#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SetMaskValueTilingData)
TILING_DATA_FIELD_DEF(int32_t, seqBs);
TILING_DATA_FIELD_DEF(int32_t, length);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SetMaskValue, SetMaskValueTilingData)
}
