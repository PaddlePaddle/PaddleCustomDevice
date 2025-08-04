#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GetStepPaddleTilingData)
  TILING_DATA_FIELD_DEF(int32_t, bsz);
  TILING_DATA_FIELD_DEF(int32_t, block_size);
  TILING_DATA_FIELD_DEF(int32_t, block_num_per_seq);
  TILING_DATA_FIELD_DEF(int32_t, max_decoder_block_num);
  TILING_DATA_FIELD_DEF(int32_t, length);
  TILING_DATA_FIELD_DEF(int32_t, pre_id_length);
  TILING_DATA_FIELD_DEF(int64_t, first_token_id);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(StepPaddle, GetStepPaddleTilingData)
}