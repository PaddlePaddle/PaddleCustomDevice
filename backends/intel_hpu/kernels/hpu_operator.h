#include <assert.h>

#include <memory>

#include "paddle/phi/backends/device_ext.h"
#include "synapse_api.h"
#include "synapse_common_types.h"
#include "utils/hpu_helper.h"

typedef std::vector<int64_t> DIMS;

#define CHKSTATUS(ERR)           \
  assert(status == synSuccess && \
         (std::string(ERR) + std::to_string(status)).c_str())

typedef std::pair<synSectionHandle, bool> sectionWithFirstIndication;
static std::unordered_map<std::string, sectionWithFirstIndication> sectionMap;

class HpuOperator {
 public:
  HpuOperator(const std::string guid) : guid_(guid) {
    synStatus status = synGraphCreate(&graphHandle_, synDeviceGaudi2);
    CHKSTATUS("synGraphCreate failed!");
  }

  void Compile() {
    synStatus status = synGraphCompile(
        &recipeHandle_, graphHandle_, (guid_ + ".recipe").c_str(), 0);
    LOG(INFO) << " synGraphCompile =" << guid_;

    CHKSTATUS("synGraphCompile failed!");
    status = synWorkspaceGetSize(&workspaceSize, recipeHandle_);
    CHKSTATUS("synWorkspaceGetSize failed!");

    LOG(INFO) << "workspace size = " << workspaceSize;
  }

  void prepareTensorInfo(synRecipeHandle recipe,
                         synLaunchTensorInfo* tensorInfo,
                         uint32_t totalNumOfTensors) {
    const char* tensorNames[totalNumOfTensors];
    uint64_t tensorIds[totalNumOfTensors];
    uint32_t i = 0;

    for (i = 0; i < totalNumOfTensors; ++i) {
      tensorNames[i] = tensorInfo[i].tensorName;
    }
    assert(synTensorRetrieveIds(
               recipe, tensorNames, tensorIds, totalNumOfTensors) ==
           synSuccess);
    for (i = 0; i < totalNumOfTensors; i++) {
      tensorInfo[i].tensorId = tensorIds[i];
    }
  }

  void Execute(C_Stream stream, std::map<std::string, uint64_t>& tensors) {
    synStatus status;
    uint64_t workspaceAddress = 0;
    LOG(INFO) << "malloc device workspace " << workspaceSize;
    // TODO
    status = synDeviceMalloc(0, workspaceSize, 0, 0, &workspaceAddress);

    CHKSTATUS("synDeviceMalloc failed!");

    std::vector<synLaunchTensorInfo> concatTensors;
    for (auto& tensor : tensors) {
      concatTensors.push_back({tensor.first.c_str(), tensor.second});
    }
    prepareTensorInfo(recipeHandle_, &concatTensors[0], concatTensors.size());
    status = synLaunch(reinterpret_cast<synStreamHandle>(stream),
                       concatTensors.data(),
                       concatTensors.size(),
                       workspaceAddress,
                       recipeHandle_,
                       0);
    CHKSTATUS("synLaunch failed!");
    status = synStreamSynchronize(reinterpret_cast<synStreamHandle>(stream));
    CHKSTATUS("synStreamSynchronize failed!");
    status = synDeviceFree(0, workspaceAddress, 0);
    CHKSTATUS("synDeviceFree failed!");
    LOG(INFO) << " =======================";
  }

  void CompileAndExecute(C_Stream stream,
                         std::map<std::string, uint64_t>& tensors) {
    Compile();
    Execute(stream, tensors);
  }

  virtual ~HpuOperator() { synGraphDestroy(graphHandle_); }

  // protected:
  synTensor createTensor(unsigned dims,
                         synDataType data_type,
                         DIMS tensor_size, /*const unsigned* tensor_size,*/
                         bool is_presist,
                         const char* name)

  {
    synStatus status;
    synTensorDescriptor desc{};
    // input
    desc.m_dataType = data_type;
    desc.m_dims = dims;
    desc.m_name = name;
    memset(desc.m_strides, 0, sizeof(desc.m_strides));

    for (unsigned i = 0; i < dims; ++i) {
      desc.m_sizes[i] = tensor_size[dims - 1 - i];
    }

    synSectionHandle sectionHandle = nullptr;
    if (is_presist) {
      const auto& section = sectionMap.find(name);
      if (section == sectionMap.end()) {
        status = synSectionCreate(&sectionHandle, 0, graphHandle_);
        assert(status == synSuccess && "Create section failed!");
      } else {
        sectionHandle = section->second.first;
      }
    }

    synTensor tensor;
    status = synTensorCreate(&tensor, &desc, sectionHandle, 0);
    assert(status == synSuccess && "Create tensor failed!");
    return tensor;
  }

  std::string guid_;
  synGraphHandle graphHandle_;
  synSectionHandle inputSectionHandle_, outputSectionHandle_;
  synRecipeHandle recipeHandle_;
  uint64_t workspaceSize;

  std::unordered_map<std::string, synRecipeHandle> recipes;
};