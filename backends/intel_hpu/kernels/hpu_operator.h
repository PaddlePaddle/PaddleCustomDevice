#include <assert.h>

#include <memory>

#include "glog/logging.h"
#include "paddle/phi/backends/device_ext.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/extension.h"
#include "synapse_api.h"
#include "synapse_common_types.h"
#include "utils/hpu_helper.h"

typedef std::vector<int64_t> DIMS;

#define CHKSTATUS(ERR)           \
  assert(status == synSuccess && \
         (std::string(ERR) + std::to_string(status)).c_str())

typedef std::pair<synSectionHandle, bool> sectionWithFirstIndication;
static std::unordered_map<std::string, sectionWithFirstIndication> sectionMap;

static uint64_t cached_orkspaceSize = 0;
static uint64_t cached_workspaceAddress = 0;

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
    uint64_t request_workspace_size = 0;
    status = synWorkspaceGetSize(&request_workspace_size, recipeHandle_);
    CHKSTATUS("synWorkspaceGetSize failed!");

    if (request_workspace_size > cached_orkspaceSize) {
      if (cached_orkspaceSize != 0) {
        status = synDeviceFree(0, cached_workspaceAddress, 0);
        LOG_IF(ERROR, status != synSuccess)
            << "[RUNTIME] synDeviceFree() failed = " << status;
      }

      cached_orkspaceSize = request_workspace_size;
      // malloc the new one
      LOG(INFO) << "malloc device workspace " << cached_orkspaceSize;
      status = synDeviceMalloc(
          0, cached_orkspaceSize, 0, 0, &cached_workspaceAddress);
      LOG_IF(ERROR, status != synSuccess)
          << "[RUNTIME] synDeviceMalloc() failed = " << status;

      CHKSTATUS("synDeviceMalloc failed!");
    }

    LOG(INFO) << "workspace size = " << cached_orkspaceSize;
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

    std::vector<synLaunchTensorInfo> concatTensors;
    for (auto& tensor : tensors) {
      concatTensors.push_back({tensor.first.c_str(), tensor.second});
    }
    prepareTensorInfo(recipeHandle_, &concatTensors[0], concatTensors.size());
    status = synLaunch(reinterpret_cast<synStreamHandle>(stream),
                       concatTensors.data(),
                       concatTensors.size(),
                       cached_workspaceAddress,
                       recipeHandle_,
                       0);

    PD_CHECK(status == synSuccess, "[RUNTIME] synLaunch() failed = %d", status);
    status = synStreamSynchronize(reinterpret_cast<synStreamHandle>(stream));
    PD_CHECK(status == synSuccess,
             "[RUNTIME] synStreamSynchronize() failed = %d",
             status);
  }

  void CompileAndExecute(C_Stream stream,
                         std::map<std::string, uint64_t>& tensors) {
    Compile();
    Execute(stream, tensors);
  }

  virtual ~HpuOperator() {
    synStatus status = synGraphDestroy(graphHandle_);
    for (size_t i = 0; i < tensors_.size(); i++) {
      status = synTensorDestroy(tensors_[i]);
      LOG_IF(ERROR, status != synSuccess)
          << "[RUNTIME] synTensorDestroy() failed = " << status;
    }
    for (size_t i = 0; i < sectons_.size(); i++) {
      status = synSectionDestroy(sectons_[i]);
      LOG_IF(ERROR, status != synSuccess)
          << "[RUNTIME] synSectionDestroy() failed = " << status;
    }
  }

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
      LOG(INFO) << "name = " << name << ", " << tensor_size[dims - 1 - i];
    }

    synSectionHandle sectionHandle = nullptr;
    if (is_presist) {
      status = synSectionCreate(&sectionHandle, 0, graphHandle_);
      LOG_IF(ERROR, status != synSuccess)
          << "[RUNTIME] synSectionCreate() failed = " << status;
      sectons_.push_back(sectionHandle);
    }

    synTensor tensor = nullptr;
    status = synTensorCreate(&tensor, &desc, sectionHandle, 0);
    LOG_IF(ERROR, status != synSuccess)
        << "[RUNTIME] synTensorCreate() failed = " << status;
    tensors_.push_back(tensor);
    return tensor;
  }

  std::string guid_;
  synGraphHandle graphHandle_;
  synRecipeHandle recipeHandle_;
  std::vector<synTensor> tensors_;
  std::vector<synSectionHandle> sectons_;
};