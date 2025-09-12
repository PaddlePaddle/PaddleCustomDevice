#pragma once

#include <mudnn.h>

using cudnnStatus_t = musa::dnn::Status;

#define CUDNN_STATUS_SUCCESS musa::dnn::Status::SUCCESS


// #define REPLACE_TO_FAKE_MUDNN_ST(__CUDNN_TYPE__)  \
//   struct mudnn##__CUDNN_TYPE__ {  \
//     mudnn##__CUNN_TYPE__(){ \
//         throw std::string(#__CUDNN_TYPE__) + std::string(" is not supported in musa"); \
//     } \ 
//   }; \
//   using cudnn##__CUDNN_TYPE__ = mudnn##__CUDNN_TYPE__; 
// 
// REPLACE_TO_FAKE_MUDNN_ST(SetCallback);                             
// REPLACE_TO_FAKE_MUDNN_ST(SetTensor4dDescriptor);                   
// REPLACE_TO_FAKE_MUDNN_ST(SetTensor4dDescriptorEx);                 
// REPLACE_TO_FAKE_MUDNN_ST(SetTensorNdDescriptor);                   
// REPLACE_TO_FAKE_MUDNN_ST(GetTensorNdDescriptor);                   
// REPLACE_TO_FAKE_MUDNN_ST(GetConvolutionNdForwardOutputDim);        
// REPLACE_TO_FAKE_MUDNN_ST(CreateTensorDescriptor);                  
// REPLACE_TO_FAKE_MUDNN_ST(DestroyTensorDescriptor);                 
// REPLACE_TO_FAKE_MUDNN_ST(CreateFilterDescriptor);                  
// REPLACE_TO_FAKE_MUDNN_ST(SetFilter4dDescriptor);                   
// REPLACE_TO_FAKE_MUDNN_ST(SetFilterNdDescriptor);                   
// REPLACE_TO_FAKE_MUDNN_ST(GetFilterNdDescriptor);                   
// REPLACE_TO_FAKE_MUDNN_ST(SetPooling2dDescriptor);                  
// REPLACE_TO_FAKE_MUDNN_ST(SetPoolingNdDescriptor);                  
// REPLACE_TO_FAKE_MUDNN_ST(GetPoolingNdDescriptor);                  
// REPLACE_TO_FAKE_MUDNN_ST(DestroyFilterDescriptor);                 
// REPLACE_TO_FAKE_MUDNN_ST(CreateConvolutionDescriptor);             
// REPLACE_TO_FAKE_MUDNN_ST(CreatePoolingDescriptor);                 
// REPLACE_TO_FAKE_MUDNN_ST(DestroyPoolingDescriptor);                
// REPLACE_TO_FAKE_MUDNN_ST(SetConvolution2dDescriptor);              
// REPLACE_TO_FAKE_MUDNN_ST(DestroyConvolutionDescriptor);            
// REPLACE_TO_FAKE_MUDNN_ST(SetConvolutionNdDescriptor);              
// REPLACE_TO_FAKE_MUDNN_ST(GetConvolutionNdDescriptor);              
// REPLACE_TO_FAKE_MUDNN_ST(DeriveBNTensorDescriptor);                
// REPLACE_TO_FAKE_MUDNN_ST(CreateSpatialTransformerDescriptor);      
// REPLACE_TO_FAKE_MUDNN_ST(SetSpatialTransformerNdDescriptor);       
// REPLACE_TO_FAKE_MUDNN_ST(DestroySpatialTransformerDescriptor);     
// REPLACE_TO_FAKE_MUDNN_ST(SpatialTfGridGeneratorForward);           
// REPLACE_TO_FAKE_MUDNN_ST(SpatialTfGridGeneratorBackward);          
// REPLACE_TO_FAKE_MUDNN_ST(SpatialTfSamplerForward);                 
// REPLACE_TO_FAKE_MUDNN_ST(SpatialTfSamplerBackward);                
// REPLACE_TO_FAKE_MUDNN_ST(Create);                                  
// REPLACE_TO_FAKE_MUDNN_ST(Destroy);                                 
// REPLACE_TO_FAKE_MUDNN_ST(SetStream);                               
// REPLACE_TO_FAKE_MUDNN_ST(ActivationForward);                       
// REPLACE_TO_FAKE_MUDNN_ST(ActivationBackward);                      
// REPLACE_TO_FAKE_MUDNN_ST(ConvolutionForward);                      
// REPLACE_TO_FAKE_MUDNN_ST(ConvolutionBackwardBias);                 
// REPLACE_TO_FAKE_MUDNN_ST(GetConvolutionForwardWorkspaceSize);      
// REPLACE_TO_FAKE_MUDNN_ST(TransformTensor);                         
// REPLACE_TO_FAKE_MUDNN_ST(PoolingForward);                          
// REPLACE_TO_FAKE_MUDNN_ST(PoolingBackward);                         
// REPLACE_TO_FAKE_MUDNN_ST(SoftmaxBackward);                         
// REPLACE_TO_FAKE_MUDNN_ST(SoftmaxForward);                          
// REPLACE_TO_FAKE_MUDNN_ST(GetVersion);                              
// REPLACE_TO_FAKE_MUDNN_ST(FindConvolutionForwardAlgorithmEx);       
// REPLACE_TO_FAKE_MUDNN_ST(FindConvolutionBackwardFilterAlgorithmEx);
// REPLACE_TO_FAKE_MUDNN_ST(FindConvolutionBackwardFilterAlgorithm);  
// REPLACE_TO_FAKE_MUDNN_ST(FindConvolutionBackwardDataAlgorithmEx);  
// REPLACE_TO_FAKE_MUDNN_ST(GetErrorString);                          
// REPLACE_TO_FAKE_MUDNN_ST(CreateDropoutDescriptor);                 
// REPLACE_TO_FAKE_MUDNN_ST(DropoutGetStatesSize);                    
// REPLACE_TO_FAKE_MUDNN_ST(SetDropoutDescriptor);                    
// REPLACE_TO_FAKE_MUDNN_ST(RestoreDropoutDescriptor);                
// REPLACE_TO_FAKE_MUDNN_ST(CreateRNNDescriptor);                     
// REPLACE_TO_FAKE_MUDNN_ST(DestroyDropoutDescriptor);                
// REPLACE_TO_FAKE_MUDNN_ST(DestroyRNNDescriptor);                    
// REPLACE_TO_FAKE_MUDNN_ST(SetTensorNdDescriptorEx);                 
// REPLACE_TO_FAKE_MUDNN_ST(AddTensor);                               
// REPLACE_TO_FAKE_MUDNN_ST(ConvolutionBackwardData);                 
// REPLACE_TO_FAKE_MUDNN_ST(ConvolutionBackwardFilter);               
// REPLACE_TO_FAKE_MUDNN_ST(GetConvolutionBackwardFilterWorkspaceSize);
// REPLACE_TO_FAKE_MUDNN_ST(GetConvolutionBackwardDataWorkspaceSize); 
// REPLACE_TO_FAKE_MUDNN_ST(BatchNormalizationForwardTraining);       
// REPLACE_TO_FAKE_MUDNN_ST(BatchNormalizationForwardInference);      
// REPLACE_TO_FAKE_MUDNN_ST(BatchNormalizationBackward);              
// REPLACE_TO_FAKE_MUDNN_ST(CreateActivationDescriptor);              
// REPLACE_TO_FAKE_MUDNN_ST(SetActivationDescriptor);                 
// REPLACE_TO_FAKE_MUDNN_ST(GetActivationDescriptor);                 
// REPLACE_TO_FAKE_MUDNN_ST(DestroyActivationDescriptor);
// 
// REPLACE_TO_FAKE_MUDNN_ST(GetConvolutionBackwardFilterAlgorithm); 
// REPLACE_TO_FAKE_MUDNN_ST(GetConvolutionForwardAlgorithm);        
// REPLACE_TO_FAKE_MUDNN_ST(GetConvolutionBackwardDataAlgorithm);   
// REPLACE_TO_FAKE_MUDNN_ST(SetRNNDescriptor);
// 
// REPLACE_TO_FAKE_MUDNN_ST(SetConvolutionGroupCount);                
// REPLACE_TO_FAKE_MUDNN_ST(SetConvolutionMathType);                  
// REPLACE_TO_FAKE_MUDNN_ST(ConvolutionBiasActivationForward);        
// REPLACE_TO_FAKE_MUDNN_ST(CreateCTCLossDescriptor);                 
// REPLACE_TO_FAKE_MUDNN_ST(DestroyCTCLossDescriptor);                
// REPLACE_TO_FAKE_MUDNN_ST(GetCTCLossDescriptor);                    
// REPLACE_TO_FAKE_MUDNN_ST(SetCTCLossDescriptor);                    
// REPLACE_TO_FAKE_MUDNN_ST(GetCTCLossWorkspaceSize);                 
// REPLACE_TO_FAKE_MUDNN_ST(CTCLoss);                                 
// REPLACE_TO_FAKE_MUDNN_ST(GetConvolutionBackwardDataAlgorithm_v7);  
// REPLACE_TO_FAKE_MUDNN_ST(GetConvolutionBackwardFilterAlgorithm_v7);
// REPLACE_TO_FAKE_MUDNN_ST(GetConvolutionForwardAlgorithm_v7);       
// REPLACE_TO_FAKE_MUDNN_ST(GetConvolutionBackwardFilterAlgorithmMaxCount);
// 
// REPLACE_TO_FAKE_MUDNN_ST(CreateRNNDataDescriptor);  
// REPLACE_TO_FAKE_MUDNN_ST(DestroyRNNDataDescriptor); 
// REPLACE_TO_FAKE_MUDNN_ST(SetRNNDataDescriptor);
// 
// REPLACE_TO_FAKE_MUDNN_ST(GetBatchNormalizationForwardTrainingExWorkspaceSize); 
// REPLACE_TO_FAKE_MUDNN_ST(BatchNormalizationForwardTrainingEx);                 
// REPLACE_TO_FAKE_MUDNN_ST(GetBatchNormalizationBackwardExWorkspaceSize);        
// REPLACE_TO_FAKE_MUDNN_ST(BatchNormalizationBackwardEx);                        
// REPLACE_TO_FAKE_MUDNN_ST(GetBatchNormalizationTrainingExReserveSpaceSize);
// 
// REPLACE_TO_FAKE_MUDNN_ST(SetRNNDescriptor_v8);                  
// REPLACE_TO_FAKE_MUDNN_ST(CreateFusedOpsPlan);                   
// REPLACE_TO_FAKE_MUDNN_ST(CreateFusedOpsConstParamPack);         
// REPLACE_TO_FAKE_MUDNN_ST(CreateFusedOpsVariantParamPack);       
// REPLACE_TO_FAKE_MUDNN_ST(DestroyFusedOpsPlan);                  
// REPLACE_TO_FAKE_MUDNN_ST(DestroyFusedOpsConstParamPack);        
// REPLACE_TO_FAKE_MUDNN_ST(DestroyFusedOpsVariantParamPack);      
// REPLACE_TO_FAKE_MUDNN_ST(FusedOpsExecute);                      
// REPLACE_TO_FAKE_MUDNN_ST(SetFusedOpsConstParamPackAttribute);   
// REPLACE_TO_FAKE_MUDNN_ST(SetFusedOpsVariantParamPackAttribute); 
// REPLACE_TO_FAKE_MUDNN_ST(MakeFusedOpsPlan);
// 
// REPLACE_TO_FAKE_MUDNN_ST(BackendCreateDescriptor);    
// REPLACE_TO_FAKE_MUDNN_ST(BackendDestroyDescriptor);   
// REPLACE_TO_FAKE_MUDNN_ST(BackendExecute);             
// REPLACE_TO_FAKE_MUDNN_ST(BackendFinalize);            
// REPLACE_TO_FAKE_MUDNN_ST(BackendGetAttribute);        
// REPLACE_TO_FAKE_MUDNN_ST(BackendSetAttribute);        
// REPLACE_TO_FAKE_MUDNN_ST(GetStream);                  
// REPLACE_TO_FAKE_MUDNN_ST(ReorderFilterAndBias);
// 
// REPLACE_TO_FAKE_MUDNN_ST(GetRNNParamsSize);          
// REPLACE_TO_FAKE_MUDNN_ST(GetRNNWorkspaceSize);       
// REPLACE_TO_FAKE_MUDNN_ST(GetRNNTrainingReserveSize); 
// REPLACE_TO_FAKE_MUDNN_ST(SetRNNDescriptor_v6);       
// REPLACE_TO_FAKE_MUDNN_ST(RNNForwardInference);       
// REPLACE_TO_FAKE_MUDNN_ST(RNNForwardTraining);        
// REPLACE_TO_FAKE_MUDNN_ST(RNNBackwardData);           
// REPLACE_TO_FAKE_MUDNN_ST(RNNBackwardWeights);
// 
// REPLACE_TO_FAKE_MUDNN_ST(SetRNNPaddingMode);      
// REPLACE_TO_FAKE_MUDNN_ST(RNNForwardInferenceEx);  
// REPLACE_TO_FAKE_MUDNN_ST(RNNForwardTrainingEx);   
// REPLACE_TO_FAKE_MUDNN_ST(RNNBackwardDataEx);      
// REPLACE_TO_FAKE_MUDNN_ST(RNNBackwardWeightsEx);
// 
// REPLACE_TO_FAKE_MUDNN_ST(GetLastErrorString);        
// REPLACE_TO_FAKE_MUDNN_ST(GetRNNWeightSpaceSize);     
// REPLACE_TO_FAKE_MUDNN_ST(GetRNNTempSpaceSizes);      
// REPLACE_TO_FAKE_MUDNN_ST(RNNForward);                
// REPLACE_TO_FAKE_MUDNN_ST(RNNBackwardData_v8);        
// REPLACE_TO_FAKE_MUDNN_ST(RNNBackwardWeights_v8);
// 