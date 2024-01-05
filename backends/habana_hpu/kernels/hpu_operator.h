#include "synapse_api.h"
#include "synapse_common_types.h"
#include "utils/hpu_helper.h"

typedef std::vector<int64_t> DIMS;

#define CHKSTATUS(errstr) assert(status == synSuccess && errstr)

// static void vector2ints(DIMS vectors, unsigned int* outs){
//   for (int i=0; i < vectors.size(); i++)
//     outs[i] = vectors[i];
// }

typedef std::pair<synSectionHandle, bool> sectionWithFirstIndication;
static std::unordered_map<std::string, sectionWithFirstIndication> sectionMap;

class HpuOperator {
    public:
    HpuOperator(const std::string guid) : guid_(guid) {
        synStatus status = synGraphCreate(&graphHandle_, synDeviceGaudi2);
        CHKSTATUS("synGraphCreate failed!");
    }
    // virtual void AddNode() = 0;
    void Compile() {
        synStatus status = synGraphCompile(&recipeHandle_, graphHandle_, (guid_ + ".recipe").c_str(), 0);
        CHKSTATUS("synGraphCompile failed!");
    }
    
    void prepareTensorInfo(synRecipeHandle      recipe,
                                        synLaunchTensorInfo* tensorInfo,
                                        uint32_t             totalNumOfTensors)
    {
        const char* tensorNames[totalNumOfTensors];
        uint64_t    tensorIds[totalNumOfTensors];
        uint32_t    i = 0;

        for (i = 0; i < totalNumOfTensors; ++i)
        {
            tensorNames[i] = tensorInfo[i].tensorName;
        }
        assert(synTensorRetrieveIds(recipe, tensorNames, tensorIds, totalNumOfTensors) == synSuccess);
        for (i = 0; i < totalNumOfTensors; i++)
        {
            tensorInfo[i].tensorId = tensorIds[i];
        }
    }

    void Execute(C_Stream stream, std::map<std::string, uint64_t> &tensors) {
        uint64_t workspaceSize = 0;
        synStatus status = synWorkspaceGetSize(&workspaceSize, recipeHandle_);
        LOG(INFO) << guid_ << " compiled recipe workspace size=" << workspaceSize;

        uint64_t workspaceAddress;
        status = synDeviceMalloc(stream->deviceId, workspaceSize, 0, 0, &workspaceAddress);
        CHKSTATUS("synDeviceMalloc failed!");

        std::vector<synLaunchTensorInfo> concatTensors;
        for (auto& tensor : tensors ) {
            concatTensors.push_back({tensor.first.c_str(), tensor.second});
        }
        prepareTensorInfo(recipeHandle_, &concatTensors[0], concatTensors.size());
        status = synLaunch(stream->computeStream,
                           concatTensors.data(),
                           concatTensors.size(),
                           workspaceAddress,
                           recipeHandle_,
                           0);
        CHKSTATUS("synLaunch failed!");
        status = synStreamSynchronize(stream->computeStream);
        CHKSTATUS("synStreamSynchronize failed!");
        status = synDeviceFree(stream->deviceId, workspaceAddress, 0);
        CHKSTATUS("synDeviceFree failed!");

    }

    void CompileAndExecute(C_Stream stream, std::map<std::string, uint64_t> &tensors) {
        Compile();
        Execute(stream, tensors);
    }

    virtual ~HpuOperator() {
        synGraphDestroy(graphHandle_);
    }

    // protected:
    synTensor createTensor(unsigned dims, synDataType data_type, DIMS tensor_size, /*const unsigned* tensor_size,*/
                       bool is_presist, const char* name)

    {
        synStatus             status;
        synTensorDescriptor   desc{};
        // input
        desc.m_dataType     = data_type;
        desc.m_dims         = dims;
        desc.m_name         = name;
        memset(desc.m_strides, 0, sizeof(desc.m_strides));

        for (unsigned i = 0; i < dims; ++i)
        {
            desc.m_sizes[i] = tensor_size[dims - 1 - i];
        }

        synSectionHandle sectionHandle = nullptr;
        if (is_presist)
        {
            const auto& section = sectionMap.find(name);
            if (section == sectionMap.end())
            {
                status = synSectionCreate(&sectionHandle, 0, graphHandle_);
                assert(status == synSuccess && "Create section failed!");
                // sectionMap[name + std::string()] = std::make_pair(sectionHandle, true);
                // sectionMap[name + std::string("_wu_out")] = std::make_pair(sectionHandle, false);
                // sectionMap[name + std::string("_out")] = std::make_pair(sectionHandle, false);
            }
            else
            {
                sectionHandle = section->second.first;
            }
        }

        synTensor     tensor;
        status = synTensorCreate(&tensor, &desc, sectionHandle, 0);
        assert(status == synSuccess && "Create tensor failed!");
        return tensor;
    }

    std::string guid_;
    synGraphHandle graphHandle_;
    synSectionHandle inputSectionHandle_, outputSectionHandle_;
    synRecipeHandle recipeHandle_;
};