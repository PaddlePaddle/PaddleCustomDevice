@ir.RegisterPass
def remove_fill_constant1p4():
    def pattern(reshape_v_tmp, slice0_q, slice1_q, slice_v):
        reshape_v_const = ir.PassDesc.OP.fill_constant()
        return reshape_without_shape(reshape_v_tmp, [slice0_q, reshape_v_const, slice1_q, slice_v])

    def replace(reshape_v_tmp, slice0_q, slice1_q, slice_v):
        return reshape_without_shape(reshape_v_tmp, [slice0_q, slice1_q, slice_v])
    return pattern, replace

@ir.RegisterPass
def remove_fill_constant1p3():
    def pattern(out, slice0_q, slice1_q):
        reshape_const = ir.PassDesc.OP.fill_constant()
        return reshape_without_shape(out, [slice0_q, slice1_q, reshape_const])

    def replace(out, slice0_q, slice1_q):
        return reshape_without_shape(out, [slice0_q, slice1_q])
    return pattern, replace

@ir.RegisterPass
def remove_fill_constant2p4():
    def pattern(reshape_v_tmp, slice0_q, slice_v):
        reshape_v_const1 = ir.PassDesc.OP.fill_constant()
        reshape_v_const2 = ir.PassDesc.OP.fill_constant()
        return reshape_without_shape(reshape_v_tmp, [slice0_q, reshape_v_const1, reshape_v_const2, slice_v])

    def replace(reshape_v_tmp, slice0_q, slice_v):
        return reshape_without_shape(reshape_v_tmp, [slice0_q, slice_v])
    return pattern, replace

@ir.RegisterPass
def remove_fill_constant2p3():
    def pattern(out, slice0_q):
        reshape_const1 = ir.PassDesc.OP.fill_constant()
        reshape_const2 = ir.PassDesc.OP.fill_constant()
        return reshape_without_shape(out, [slice0_q, reshape_const1, reshape_const2])

    def replace(out, slice0_q):
        return reshape_without_shape(out, [slice0_q])
    return pattern, replace

@ir.RegisterPass
def remove_fused_bias_residual_layernorm():
    def pattern(residual, x):
        op = ir.PassDesc.OP.remove_fused_bias_residual_layernorm
        op._outputs.pop("mean")
        op._outputs.pop("residual_out")
        op._outputs.pop("variance")
        result = op(
            residual=residual,
            x=x
        )
        return result.Output("out")[0]

    def replace(residual, x):
        return x
    return pattern, replace


@ir.RegisterPass
def remove_rebuild_padding():
    def pattern(input_ids, padding_offset, seq_lens, tmp_out):
        op = ir.PassDesc.OP.remove_rebuild_padding
        result = op(
            input_ids=input_ids,
            padding_offset=padding_offset,
            seq_lens=seq_lens,
            tmp_out=tmp_out)
        return result

    def replace(input_ids, padding_offset, seq_lens, tmp_out):
        return tmp_out  
    return pattern, replace


@ir.RegisterPass
def remove_get_padding_offset():
    def pattern(cum_offsets, input_ids, seq_len, token_num):
        op = ir.PassDesc.OP.remove_get_padding_offset
        result = op(
            cum_offsets=cum_offsets,
            input_ids=input_ids,
            seq_len=seq_len,
            token_num=token_num)
        return result.Output("x_remove_padding")[0], result.Output("cum_offsets_out")[0], result.Output("padding_offset")[0]

    def replace(cum_offsets, input_ids, seq_len, token_num):
        return cum_offsets, seq_len, input_ids
    return pattern, replace
