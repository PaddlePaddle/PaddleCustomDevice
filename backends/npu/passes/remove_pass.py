import paddle
from paddle.incubate.passes import ir

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
        op = ir.PassDesc.OP.fused_bias_residual_layernorm
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
        op = ir.PassDesc.OP.rebuild_padding
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
        op = ir.PassDesc.OP.get_padding_offset
        result = op(
            cum_offsets=cum_offsets,
            input_ids=input_ids,
            seq_len=seq_len,
            token_num=token_num)
        return result.Output("x_remove_padding")[0], result.Output("cum_offsets_out")[0], result.Output("padding_offset")[0]

    def replace(cum_offsets, input_ids, seq_len, token_num):
        return input_ids, seq_len, cum_offsets
    return pattern, replace

@ir.RegisterPass
def remove_get_token_penalty_multi_scores():
    def pattern(pre_ids, logits, penalty_scores, frequency_scores, presence_scores, cur_len, min_len, eos_token_id):
        op = ir.PassDesc.OP.get_token_penalty_multi_scores
        result = op(
            pre_ids=pre_ids,
            logits=logits,
            penalty_scores=penalty_scores,
            frequency_scores=frequency_scores,
            presence_scores=presence_scores,
            cur_len=cur_len,
            min_len=min_len,
            eos_token_id=eos_token_id)
        return result

    def replace(pre_ids, logits, penalty_scores, frequency_scores, presence_scores, cur_len, min_len, eos_token_id):
        return logits
    return pattern, replace

@ir.RegisterPass
def save_with_output_delay_pass_1st():
    def pattern(topk_ids, batch_idx, step_idx):    
        save_with_output = ir.PassDesc.OP.save_with_output(x=topk_ids, batch_idx=batch_idx, step_idx=step_idx)
        
        scale = ir.PassDesc.OP.scale(X=step_idx)
        
        return save_with_output, scale

    def replace(topk_ids, batch_idx, step_idx):
       
        # save_with_output = ir.PassDesc.OP.save_with_output(x=topk_ids, batch_idx=batch_idx, step_idx=step_idx)
        save_with_output = ir.PassDesc.OP.save_with_output_delay(x=topk_ids, batch_idx=batch_idx, step_idx=step_idx, no_use=step_idx)
        save_with_output.Attr("file_path").MappedPattern(op="save_with_output", name="file_path", index=0)
        save_with_output.Attr("rank_id").MappedPattern(op="save_with_output", name="rank_id", index=0)
        save_with_output_0 = save_with_output.Output("out")[0]
        save_with_output_1 = save_with_output.Output("no_use_out")[0]
        
        return save_with_output_0, save_with_output_1

    return pattern, replace

@ir.RegisterPass
def save_with_output_delay_pass_2nd():
    def pattern(topk_ids, batch_idx, step_idx, less_x, less_y):    
        save_with_output = ir.PassDesc.OP.save_with_output(x=topk_ids, batch_idx=batch_idx, step_idx=step_idx)

        less_than = ir.PassDesc.OP.less_than(X=less_x, Y=less_y)
        assign = ir.PassDesc.OP.assign(X=less_than)
        
        return save_with_output, assign

    def replace(topk_ids, batch_idx, step_idx, less_x, less_y):
        less_than = ir.PassDesc.OP.less_than(X=less_x, Y=less_y)
        less_than.Attr("axis").MappedPattern(op="less_than", name="axis", index=0)
        less_than.Attr("force_cpu").MappedPattern(op="less_than", name="force_cpu", index=0)
        
        assign = ir.PassDesc.OP.assign(X=less_than)
        
        # save_with_output = ir.PassDesc.OP.save_with_output(x=topk_ids, batch_idx=batch_idx, step_idx=step_idx)
        save_with_output = ir.PassDesc.OP.save_with_output_delay(x=topk_ids, batch_idx=batch_idx, step_idx=step_idx, no_use=assign)
        save_with_output.Attr("file_path").MappedPattern(op="save_with_output", name="file_path", index=0)
        save_with_output.Attr("rank_id").MappedPattern(op="save_with_output", name="rank_id", index=0)
        save_with_output_0 = save_with_output.Output("out")[0]
        save_with_output_1 = save_with_output.Output("no_use_out")[0]
        
        return save_with_output_0, save_with_output_1

    return pattern, replace

@ir.RegisterPass
def remove_get_max_len():
    def pattern(seq_lens_decoder, seq_lens_encoder):
        get_max_len_out = ir.PassDesc.OP.get_max_len(
            seq_lens_decoder=seq_lens_decoder, seq_lens_encoder=seq_lens_encoder)
        seq_lens_decoder_out = get_max_len_out.Output("seq_lens_decoder_out")[0]
        seq_lens_encoder_out = get_max_len_out.Output("seq_lens_encoder_out")[0]
        return seq_lens_decoder_out, seq_lens_encoder_out

    def replace(seq_lens_decoder, seq_lens_encoder):
        return seq_lens_decoder, seq_lens_encoder
    return pattern, replace