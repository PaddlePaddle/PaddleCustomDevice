#include <musparse.h>

using cusparseStatus_t = musparseStatus_t;

#define CUSPARSE_STATUS_SUCCESS MUSPARSE_STATUS_SUCCESS
#define cudaGetErrorString musaGetErrorString 