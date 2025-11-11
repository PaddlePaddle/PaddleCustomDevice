// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "utils/utils.h"

#include <filesystem>
#include <functional>

FLAGS_DEFINE_string(intel_hpu_recipe_cache_config,
                    "",
                    "configuration of recipe cache");
FLAGS_DEFINE_bool(intel_hpu_recipe_cache_debug,
                  false,
                  "recipe cache debug log");
FLAGS_DEFINE_int32(intel_hpu_recipe_cache_num,
                   10240,
                   "recipe cache queue number");

#define RECIPE_SUFFIX ".recipe"

std::string ShowErrorMsg(synStatus s) {
  char msg[STATUS_DESCRIPTION_MAX_SIZE] = {0};

  synStatusGetBriefDescription(s, msg, STATUS_DESCRIPTION_MAX_SIZE);
  return std::string(msg);
}

namespace fs = std::filesystem;

bool file_exists(std::string const& file) { return fs::exists(file); }

std::string hashToHexString(size_t hashValue) {
  std::ostringstream oss;
  oss << std::hex << std::setw(16) << std::setfill('0') << hashValue;
  return oss.str();
}

std::string recipe_file_path(std::string const& path,
                             const std::string& cache_id) {
  std::hash<std::string> hasher;
  size_t hashValue = hasher(cache_id);
  return path + "/" + hashToHexString(hashValue) + RECIPE_SUFFIX;
}

void delete_recipes(std::string recipe_path) {
  fs::path directoryPath = recipe_path;

  try {
    if (!fs::exists(directoryPath)) {
      LOG_IF(ERROR, FLAGS_intel_hpu_recipe_cache_debug)
          << "path not exists: " << directoryPath << std::endl;
      return;
    }

    if (!fs::is_directory(directoryPath)) {
      LOG_IF(ERROR, FLAGS_intel_hpu_recipe_cache_debug)
          << "not a directory: " << directoryPath << std::endl;
      return;
    }

    for (const auto& entry : fs::directory_iterator(directoryPath)) {
      if (fs::is_regular_file(entry) &&
          entry.path().extension() == RECIPE_SUFFIX) {
        LOG_IF(INFO, FLAGS_intel_hpu_recipe_cache_debug)
            << "deleting file: " << entry.path() << std::endl;
        fs::remove(entry.path());
      }
    }

    LOG_IF(INFO, FLAGS_intel_hpu_recipe_cache_debug)
        << "delete done." << std::endl;
  } catch (const fs::filesystem_error& e) {
    LOG_IF(ERROR, FLAGS_intel_hpu_recipe_cache_debug)
        << "filesystem error: " << e.what() << std::endl;
    return;
  } catch (const std::exception& e) {
    LOG_IF(ERROR, FLAGS_intel_hpu_recipe_cache_debug)
        << "filesystem unknown error: " << e.what() << std::endl;
    return;
  }
}

RecipeCache::RecipeCache() {
  cache_path_ = "";
  cache_delete_ = false;
  cache_size_mb_ = 0;

  auto cache_config = FLAGS_intel_hpu_recipe_cache_config;
  if (!cache_config.empty()) {
    std::istringstream iss(cache_config);
    std::string param;
    std::vector<std::string> params;
    while (std::getline(iss, param, ',')) {
      params.push_back(param);
    }
    if (params.size() > 0) cache_path_ = params[0];
    if (params.size() > 1) cache_delete_ = (params[1] == "true");
    if (params.size() > 2) cache_size_mb_ = std::stoi(params[2]);

    LOG_IF(INFO, FLAGS_intel_hpu_recipe_cache_debug)
        << "cache_config: " << cache_config << std::endl
        << "cache_path: " << cache_path_ << ", cache_delete: " << cache_delete_
        << ", cache_size_mb: " << cache_size_mb_ << std::endl;
    if (cache_delete_) delete_recipes(cache_path_);
  }
}

std::string get_local_time() {
  timeval tv;
  gettimeofday(&tv, nullptr);

  char buf[30];
  strftime(buf, sizeof(buf), "%F %T", localtime(&tv.tv_sec));

  std::stringstream ss;
  ss << buf << "." << tv.tv_usec;
  return ss.str();
}

void RecipeCache::store(std::string cache_id, synRecipeHandle recipe_handle) {
  if (isEnabled() == false) return;

  auto recipe_path = recipe_file_path(cache_path_, cache_id);
  if (file_exists(recipe_path)) return;
  LOG_IF(INFO, FLAGS_intel_hpu_recipe_cache_debug)
      << "save recipe: key=" << cache_id << ", file=" << recipe_path
      << std::endl;
  std::string tmp_recipe_path = recipe_path + get_local_time();
  auto status = synRecipeSerialize(recipe_handle, tmp_recipe_path.c_str());
  PD_CHECK(status == synSuccess,
           "[Recipe] serialize recipe failed, ",
           ShowErrorMsg(status));
  rename(tmp_recipe_path.c_str(), recipe_path.c_str());
  LOG_IF(INFO, FLAGS_intel_hpu_recipe_cache_debug)
      << "save recipe " << recipe_path << " done\n";
  return;
}

synRecipeHandle RecipeCache::lookup(std::string cache_id) {
  if (isEnabled() == false) return nullptr;

  auto recipe_path = recipe_file_path(cache_path_, cache_id);

  if (file_exists(recipe_path)) {
    synRecipeHandle recipeHandle = nullptr;
    auto status = synRecipeDeSerialize(&recipeHandle, recipe_path.c_str());

    if (status != synSuccess) {
      LOG_IF(WARNING, FLAGS_intel_hpu_recipe_cache_debug)
          << "Failed to deserialize recipe: " << ShowErrorMsg(status);
      return nullptr;
    }
    LOG_IF(INFO, FLAGS_intel_hpu_recipe_cache_debug)
        << "load recipe from cache: key=" << cache_id
        << ", file=" << recipe_path << std::endl;
    return recipeHandle;
  } else {
    LOG_IF(INFO, FLAGS_intel_hpu_recipe_cache_debug)
        << "Not find recipe from cache: " << recipe_path;
    return nullptr;
  }
}

void RecipeCache::flush() {}

template <class KEY_T, class VAL_T>
LRUCache<KEY_T, VAL_T>::LRUCache() {
  cache_num = FLAGS_intel_hpu_recipe_cache_num;
}

template <class KEY_T, class VAL_T>
void LRUCache<KEY_T, VAL_T>::clean(void) {
  while (item_map.size() > cache_num) {
    auto last_it = item_list.end();
    last_it--;
    item_map.erase(last_it->first);
    item_list.pop_back();
  }
}

template <class KEY_T, class VAL_T>
void LRUCache<KEY_T, VAL_T>::put(const KEY_T& key, const VAL_T& val) {
  auto it = item_map.find(key);
  if (it != item_map.end()) {
    item_list.erase(it->second);
    item_map.erase(it);
  }
  item_list.push_front(make_pair(key, val));
  item_map.insert(make_pair(key, item_list.begin()));
  clean();
  recipe_cache.store(key, val);
}

template <class KEY_T, class VAL_T>
VAL_T LRUCache<KEY_T, VAL_T>::get(const KEY_T& key) {
  auto it = item_map.find(key);
  if (it != item_map.end()) {
    item_list.splice(item_list.begin(), item_list, it->second);
    return it->second->second;
  }

  return nullptr;
}

template <class KEY_T, class VAL_T>
VAL_T LRUCache<KEY_T, VAL_T>::search(const KEY_T& key) {
  // search recipe cache if it is set
  auto recipe = recipe_cache.lookup(key);
  if (recipe) {
    item_list.push_front(make_pair(key, recipe));
    item_map.insert(make_pair(key, item_list.begin()));
    clean();
  }
  return recipe;
}

template class LRUCache<std::string, synRecipeHandle>;
