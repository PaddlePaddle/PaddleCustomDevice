import paddle
from paddle.incubate.passes import ir

# TODO wait for Custom_op
def llama_paralle_cached_layer_adaptor(lookup, in_scale, ln_bias, ffn_ln_bias, qkv_weight, qkv_bias, qkv_out_scale, proj_weight, out_proj_bias, out_scale, ffn_in_scale, ffn1_weight, ffn1_bias, ffn1_out_scale, ffn2_weight, ffn2_bias, ffn2_out_scale, cos_table, sin_table,
                                       att_mask, key_cache, value_cache, seq_lens_decoder, seq_lens_encoder, block_tables):
    llama_decoder_layer_parallel_op = ir.PassDesc.OP.llama_blockattn_layer_parallel
    llama_decoder_layer_parallel_op._outputs = {}
    llama_decoder_layer_parallel_op(
            Hidden=lookup,
            NormWeight=in_scale,
            InputNormBeta=ln_bias,
            SelfNormBeta=ffn_ln_bias,
            QKVMixWeight=qkv_weight,
            QKVDeqBias=qkv_bias,
            QKVDeqScale=qkv_out_scale,
            SelfOutLinearWeight=proj_weight,
            SelfOutLinearDeqBias=out_proj_bias,
            SelfOutLinearDeqScale=out_scale,
            SelfOutNormWeight=ffn_in_scale,
            MlpGateUpWeight=ffn1_weight,
            MlpDeqBias=ffn1_bias,
            MlpDeqScale=ffn1_out_scale,
            MlpDownWeight=ffn2_weight,
            MlpDownDeqBias=ffn2_bias,
            MlpDownDeqScale=ffn2_out_scale,
            CosTable=cos_table,
            SinTable=sin_table,
            AttentionMask=att_mask,
            Cache_K=key_cache,
            Cache_V=value_cache,
            DecoderSeqLength=seq_lens_decoder,
            EncoderSeqLength=seq_lens_encoder,
            BlockTables=block_tables,
    )

    outs_name = [paddle.base.unique_name.generate('llama_decoder_layer_parallel') for i in range(3)] # 3 outputs
    print(outs_name)
    llama_decoder_layer_parallel_op._desc.set_output("Out", [outs_name[0]])

    llama_decoder_layer_parallel_op.Attr("block_size").MappedPattern(op="block_multihead_attention", name="block_size", index=0)
    llama_decoder_layer_parallel_op.Attr("rmsNormEps").MappedPattern(op="rms_norm", name="epsilon", index=0)
    llama_decoder_layer_parallel_op.Attr("inputRmsNormScale").MappedPattern(op="rms_norm", name="quant_scale", index=0)
    llama_decoder_layer_parallel_op.Attr("selfRmsNormScale").MappedPattern(op="rms_norm", name="quant_scale", index=1)
    llama_decoder_layer_parallel_op.Attr("selfQuantScale").MappedPattern(op="block_multihead_attention", name="out_scale", index=0)
    llama_decoder_layer_parallel_op.Attr("mlpQuantScale").MappedPattern(op="fused_bias_act", name="quant_scale", index=0)

    block = paddle.static.default_main_program().current_block()
    results = []
    for out in outs_name:
        results.append(block.create_var(name=out))
    return results[0], results[1], results[2]

def dequant_linear(matmul_input, matmul_weight, dequant_scale):
    matmul = ir.PassDesc.OP.matmul_v2(X=matmul_input, Y=matmul_weight)
    dequant = ir.PassDesc.OP.dequant_int8(intput=matmul, out_scale=dequant_scale)
    return ir.PassDesc.OP.c_allreduce_sum(X=dequant)


@ir.RegisterPass
def llama_fuse_attention_dynamic_parallel_first():
    def pattern(lookup, in_scale, qkv_weight, proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight, ln_bias, block_tables, cu_seqlens_k, cu_seqlens_q, 
            cum_offsets, key_cache, out_shift, out_smooth, padding_offsets, qkv_bias, qkv_out_scale, seq_lens_decoder, seq_lens_encoder, seq_lens_this_time, 
            value_cache, out_scale, out_proj_bias, ffn_ln_bias, ffn1_bias, ffn1_out_scale, ffn2_shift, ffn2_smooth, ffn2_out_scale, att_mask, cos_table, sin_table):
        rms_norm_ = ir.PassDesc.OP.rms_norm(norm_bias=ln_bias, norm_weight=in_scale, x=lookup)
        qkv_matmul = ir.PassDesc.OP.matmul_v2(X=rms_norm_.Output("out"), Y=qkv_weight)
        attention_ = ir.PassDesc.OP.block_multihead_attention(
            block_tables=block_tables,
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_q=cu_seqlens_q,
            cum_offsets=cum_offsets,
            key_cache=key_cache,
            mask=att_mask,
            out_shift=out_shift,
            out_smooth=out_smooth,
            padding_offsets=padding_offsets,
            qkv=qkv_matmul,
            qkv_bias=qkv_bias,
            qkv_out_scale=qkv_out_scale,
            rope_emb=cos_table,
            seq_lens_decoder=seq_lens_decoder,
            seq_lens_encoder=seq_lens_encoder,
            seq_lens_this_time=seq_lens_this_time,
            tgt_mask=sin_table,
            value_cache=value_cache)

        allreduce_sum1 = dequant_linear(attention_.Output("fmha_out"), proj_weight, out_scale)
        ffn_rms_norm = ir.PassDesc.OP.rms_norm(bias=out_proj_bias, norm_bias=ffn_ln_bias, norm_weight=ffn_in_scale, residual=lookup, x=allreduce_sum1)
        ffn_matmul1 = ir.PassDesc.OP.matmul_v2(X=ffn_rms_norm.Output("out"), Y=ffn1_weight)
        fused_bias_act_ = ir.PassDesc.OP.fused_bias_act(bias=ffn1_bias, dequant_scales=ffn1_out_scale, shift=ffn2_shift, smooth=ffn2_smooth, x=ffn_matmul1)
        allreduce_sum2 = dequant_linear(fused_bias_act_, ffn2_weight, ffn2_out_scale)

        return ffn_rms_norm.Output("residual_out")[0], allreduce_sum2

    def replace(lookup, in_scale, qkv_weight, proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight, ln_bias, block_tables, cu_seqlens_k, cu_seqlens_q, 
            cum_offsets, key_cache, out_shift, out_smooth, padding_offsets, qkv_bias, qkv_out_scale, seq_lens_decoder, seq_lens_encoder, seq_lens_this_time, 
            value_cache, out_scale, out_proj_bias, ffn_ln_bias, ffn1_bias, ffn1_out_scale, ffn2_shift, ffn2_smooth, ffn2_out_scale, att_mask, cos_table, sin_table):
        result = llama_paralle_cached_layer_adaptor(lookup, in_scale, ln_bias, ffn_ln_bias, qkv_weight, qkv_bias, qkv_out_scale, proj_weight, out_proj_bias, out_scale, ffn_in_scale, ffn1_weight, ffn1_bias, ffn1_out_scale, ffn2_weight, ln_bias, ffn2_out_scale, cos_table, sin_table,
                                                    att_mask, key_cache, value_cache, seq_lens_decoder, seq_lens_encoder, block_tables)

        return result[2], result[0]
        # add1 = ir.PassDesc.OP.my_add_n(X=lookup, Y=cu_seqlens_k, Z=cu_seqlens_q)
        # add2 = ir.PassDesc.OP.my_add_n(X=lookup, Y=add1, Z=cu_seqlens_q)
        # return add1, add2

    return pattern, replace

@ir.RegisterPass
def llama_fuse_attention_dynamic_parallel_others():
    def pattern(in_scale, rms_norm_residual, matmul_, qkv_weight, proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight, ln_bias, block_tables, cu_seqlens_k, cu_seqlens_q, 
            cum_offsets, key_cache, out_shift, out_smooth, padding_offsets, qkv_bias, qkv_out_scale, seq_lens_decoder, seq_lens_encoder, seq_lens_this_time, 
            value_cache, out_scale, out_proj_bias, ffn_ln_bias, ffn1_bias, ffn1_out_scale, ffn2_shift, ffn2_smooth, ffn2_out_scale, att_mask, cos_table, sin_table):
        rms_norm_ = ir.PassDesc.OP.rms_norm(norm_bias=ln_bias, norm_weight=in_scale, residual=rms_norm_residual, x=matmul_)
        qkv_matmul = ir.PassDesc.OP.matmul_v2(X=rms_norm_.Output("out"), Y=qkv_weight)
        attention_ = ir.PassDesc.OP.block_multihead_attention(
            block_tables=block_tables,
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_q=cu_seqlens_q,
            cum_offsets=cum_offsets,
            key_cache=key_cache,
            mask=att_mask,
            out_shift=out_shift,
            out_smooth=out_smooth,
            padding_offsets=padding_offsets,
            qkv=qkv_matmul,
            qkv_bias=qkv_bias,
            qkv_out_scale=qkv_out_scale,
            rope_emb=cos_table,
            seq_lens_decoder=seq_lens_decoder,
            seq_lens_encoder=seq_lens_encoder,
            seq_lens_this_time=seq_lens_this_time,
            tgt_mask=sin_table,
            value_cache=value_cache)

        allreduce_sum1 = dequant_linear(attention_.Output("fmha_out"), proj_weight, out_scale)
        ffn_rms_norm = ir.PassDesc.OP.rms_norm(bias=out_proj_bias, norm_bias=ffn_ln_bias, norm_weight=ffn_in_scale, residual=rms_norm_.Output("residual_out"), x=allreduce_sum1)
        ffn_matmul1 = ir.PassDesc.OP.matmul_v2(X=ffn_rms_norm.Output("out"), Y=ffn1_weight)
        fused_bias_act_ = ir.PassDesc.OP.fused_bias_act(bias=ffn1_bias, dequant_scales=ffn1_out_scale, shift=ffn2_shift, smooth=ffn2_smooth, x=ffn_matmul1)
        allreduce_sum2 = dequant_linear(fused_bias_act_, ffn2_weight, ffn2_out_scale)

        return ffn_rms_norm.Output("residual_out")[0], allreduce_sum2

    def replace(in_scale, rms_norm_residual, matmul_, qkv_weight, proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight, ln_bias, block_tables, cu_seqlens_k, cu_seqlens_q, 
            cum_offsets, key_cache, out_shift, out_smooth, padding_offsets, qkv_bias, qkv_out_scale, seq_lens_decoder, seq_lens_encoder, seq_lens_this_time, 
            value_cache, out_scale, out_proj_bias, ffn_ln_bias, ffn1_bias, ffn1_out_scale, ffn2_shift, ffn2_smooth, ffn2_out_scale, att_mask, cos_table, sin_table):
        result = llama_paralle_cached_layer_adaptor(matmul_, in_scale, ln_bias, ffn_ln_bias, qkv_weight, qkv_bias, qkv_out_scale, proj_weight, out_proj_bias, out_scale, ffn_in_scale, ffn1_weight, ffn1_bias, ffn1_out_scale, ffn2_weight, ln_bias, ffn2_out_scale, cos_table, sin_table,
                                                    att_mask, key_cache, value_cache, seq_lens_decoder, seq_lens_encoder, block_tables)
        return result[2], result[0]
        # add1 = ir.PassDesc.OP.my_add_n(X=matmul_, Y=rms_norm_residual, Z=cu_seqlens_q)
        # add2 = ir.PassDesc.OP.my_add_n(X=matmul_, Y=add1, Z=cu_seqlens_q)        
        # return add1, add2

    return pattern, replace

@ir.RegisterPass
def llama_fuse_attention_dynamic_parallel_end():
    def pattern(in_scale, rms_norm_residual, matmul_, qkv_weight, proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight, ln_bias, block_tables, cu_seqlens_k, cu_seqlens_q, 
            cum_offsets, key_cache, out_shift, out_smooth, padding_offsets, qkv_bias, qkv_out_scale, seq_lens_decoder, seq_lens_encoder, seq_lens_this_time, 
            value_cache, out_scale, out_proj_bias, ffn_ln_bias, ffn1_bias, ffn1_out_scale, ffn2_shift, ffn2_smooth, ffn2_out_scale, ffn2_bias, att_mask, cos_table, sin_table):
        rms_norm_ = ir.PassDesc.OP.rms_norm(norm_bias=ln_bias, norm_weight=in_scale, residual=rms_norm_residual, x=matmul_)
        qkv_matmul = ir.PassDesc.OP.matmul_v2(X=rms_norm_.Output("out"), Y=qkv_weight)
        attention_ = ir.PassDesc.OP.block_multihead_attention(
            block_tables=block_tables,
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_q=cu_seqlens_q,
            cum_offsets=cum_offsets,
            key_cache=key_cache,
            mask=att_mask,
            out_shift=out_shift,
            out_smooth=out_smooth,
            padding_offsets=padding_offsets,
            qkv=qkv_matmul,
            qkv_bias=qkv_bias,
            qkv_out_scale=qkv_out_scale,
            rope_emb=cos_table,
            seq_lens_decoder=seq_lens_decoder,
            seq_lens_encoder=seq_lens_encoder,
            seq_lens_this_time=seq_lens_this_time,
            tgt_mask=sin_table,
            value_cache=value_cache)

        allreduce_sum1 = dequant_linear(attention_.Output("fmha_out"), proj_weight, out_scale)
        ffn_rms_norm = ir.PassDesc.OP.rms_norm(bias=out_proj_bias, norm_bias=ffn_ln_bias, norm_weight=ffn_in_scale, residual=rms_norm_.Output("residual_out"), x=allreduce_sum1)
        ffn_matmul1 = ir.PassDesc.OP.matmul_v2(X=ffn_rms_norm.Output("out"), Y=ffn1_weight)        
        fused_bias_act_ = ir.PassDesc.OP.fused_bias_act(bias=ffn1_bias, dequant_scales=ffn1_out_scale, shift=ffn2_shift, smooth=ffn2_smooth, x=ffn_matmul1)
        allreduce_sum2 = dequant_linear(fused_bias_act_, ffn2_weight, ffn2_out_scale)
        fused_bias_residual_layernorm_ = ir.PassDesc.OP.fused_bias_residual_layernorm(bias=ffn2_bias, residual=ffn_rms_norm.Output("residual_out")[0], x=allreduce_sum2)
        
        return fused_bias_residual_layernorm_.Output("out")[0]
        
    def replace(in_scale, rms_norm_residual, matmul_, qkv_weight, proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight, ln_bias, block_tables, cu_seqlens_k, cu_seqlens_q, 
            cum_offsets, key_cache, out_shift, out_smooth, padding_offsets, qkv_bias, qkv_out_scale, seq_lens_decoder, seq_lens_encoder, seq_lens_this_time, 
            value_cache, out_scale, out_proj_bias, ffn_ln_bias, ffn1_bias, ffn1_out_scale, ffn2_shift, ffn2_smooth, ffn2_out_scale, ffn2_bias, att_mask, cos_table, sin_table):
        result = llama_paralle_cached_layer_adaptor(matmul_, in_scale, ln_bias, ffn_ln_bias, qkv_weight, qkv_bias, qkv_out_scale, proj_weight, out_proj_bias, out_scale, ffn_in_scale, ffn1_weight, ffn1_bias, ffn1_out_scale, ffn2_weight, ffn2_bias, ffn2_out_scale, cos_table, sin_table,
                                                    att_mask, key_cache, value_cache, seq_lens_decoder, seq_lens_encoder, block_tables)
        return result[0]
        # add1 = ir.PassDesc.OP.my_add_n(X=matmul_, Y=rms_norm_residual, Z=cu_seqlens_q)  
        # return add1
    return pattern, replace

@ir.RegisterPass
def remove_fused_bias_residual_layernorm_quant():
    def pattern(bias, residual, x):
        op = ir.PassDesc.OP.fused_bias_residual_layernorm
        op._outputs.pop("mean")
        op._outputs.pop("residual_out")
        op._outputs.pop("variance")
        result = op(
            bias=bias,
            residual=residual,
            x=x
        )
        return result.Output("out")[0]

    def replace(bias, residual, x):
        return x
    return pattern, replace

@ir.RegisterPass
def llama_lmhead_quant():
    def pattern(x, norm_weight, matmul_weight):
        rms_norm_ = ir.PassDesc.OP.rms_norm(x=x, norm_weight=norm_weight)
        slice_ = ir.PassDesc.OP.slice(Input=rms_norm_.Output("out"))
        identity_ = ir.PassDesc.OP.c_identity(X=rms_norm_.Output("out"))
        matmul_ = ir.PassDesc.OP.matmul_v2(X=identity_, Y=matmul_weight)
        return matmul_

    def replace(x, norm_weight, matmul_weight):
        llama_lmhead = ir.PassDesc.OP.llama_lmhead(
            Hidden=x,
            NormWeight=norm_weight,
            MatmulWeight=matmul_weight,
        )

        llama_lmhead.Attr("rmsNormEps").MappedPattern(op="rms_norm", name="epsilon", index=0)
        llama_lmhead.Attr("transpose").MappedPattern(op="matmul_v2", name="trans_y", index=0)

        return llama_lmhead

    return pattern, replace

def llama_paralle_cached_layer_adaptor_65B(lookup, in_scale, qkv_weight, qkv_out_scale, proj_weight, out_shift, out_smooth, out_scale, ffn_in_scale, ffn1_weight, ffn1_out_scale, ffn2_weight, ffn2_shift, ffn2_smooth, ffn2_out_scale, cos_table, sin_table,
                                       att_mask, key_cache, value_cache, seq_lens_decoder, seq_lens_encoder, block_tables, cache_k_dequant_scales, cache_k_quant_scales, cache_v_dequant_scales, cache_v_quant_scales):
    llama_decoder_layer_parallel_op = ir.PassDesc.OP.llama_blockattn_layer_parallel
    llama_decoder_layer_parallel_op._outputs = {}
    llama_decoder_layer_parallel_op(
           Hidden=lookup,
            NormWeight=in_scale,
            QKVMixWeight=qkv_weight,
            QKVDeqScale=qkv_out_scale,
            SelfOutLinearWeight=proj_weight,
            SelfOutLinearShift=out_shift,
            SelfOutLinearSmooth=out_smooth,
            SelfOutLinearDeqScale=out_scale,
            SelfOutNormWeight=ffn_in_scale,
            MlpGateUpWeight=ffn1_weight,
            MlpDeqScale=ffn1_out_scale,
            MlpDownWeight=ffn2_weight,
            MlpDownShift=ffn2_shift,
            MlpDownSmooth=ffn2_smooth,
            MlpDownDeqScale=ffn2_out_scale,
            CosTable=cos_table,
            SinTable=sin_table,
            AttentionMask=att_mask,
            Cache_K=key_cache,
            Cache_V=value_cache,
            DecoderSeqLength=seq_lens_decoder,
            EncoderSeqLength=seq_lens_encoder,
            BlockTables=block_tables,
            CacheKQuantScales=cache_k_quant_scales,
            CacheKDequantScales=cache_k_dequant_scales,
            CacheVQuantScales=cache_v_quant_scales,
            CacheVDequantScales=cache_v_dequant_scales,
    )

    outs_name = [paddle.base.unique_name.generate('llama_decoder_layer_parallel') for i in range(3)] # 3 outputs
    print(outs_name)
    llama_decoder_layer_parallel_op._desc.set_output("Out", [outs_name[0]])

    llama_decoder_layer_parallel_op.Attr("block_size").MappedPattern(op="block_multihead_attention", name="block_size", index=0)
    llama_decoder_layer_parallel_op.Attr("rmsNormEps").MappedPattern(op="rms_norm", name="epsilon", index=0)
    llama_decoder_layer_parallel_op.Attr("inputRmsNormScale").MappedPattern(op="rms_norm", name="quant_scale", index=0)
    llama_decoder_layer_parallel_op.Attr("selfRmsNormScale").MappedPattern(op="rms_norm", name="quant_scale", index=1)
    llama_decoder_layer_parallel_op.Attr("selfQuantScale").MappedPattern(op="block_multihead_attention", name="out_scale", index=0)
    llama_decoder_layer_parallel_op.Attr("mlpQuantScale").MappedPattern(op="fused_bias_act", name="quant_scale", index=0)

    block = paddle.static.default_main_program().current_block()
    results = []
    for out in outs_name:
        results.append(block.create_var(name=out))
    return results[0], results[1], results[2]

def dequant_linear(matmul_input, matmul_weight, dequant_scale):
    matmul = ir.PassDesc.OP.matmul_v2(X=matmul_input, Y=matmul_weight)
    dequant = ir.PassDesc.OP.dequant_int8(intput=matmul, out_scale=dequant_scale)
    return ir.PassDesc.OP.c_allreduce_sum(X=dequant)


@ir.RegisterPass
def llama_fuse_attention_dynamic_parallel_first_65B():
    def pattern(lookup, in_scale, qkv_weight, proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight, block_tables, cu_seqlens_k, cu_seqlens_q, 
            cum_offsets, key_cache, out_shift, out_smooth, padding_offsets, qkv_out_scale, seq_lens_decoder, seq_lens_encoder, seq_lens_this_time, 
            value_cache, out_scale, ffn1_out_scale, ffn2_shift, ffn2_smooth, ffn2_out_scale, att_mask, cos_table, sin_table,
            cache_k_dequant_scales, cache_k_quant_scales, cache_v_dequant_scales, cache_v_quant_scales):
        rms_norm_ = ir.PassDesc.OP.rms_norm(norm_weight=in_scale, x=lookup)
        qkv_matmul = ir.PassDesc.OP.matmul_v2(X=rms_norm_.Output("out"), Y=qkv_weight)
        attention_ = ir.PassDesc.OP.block_multihead_attention(
            block_tables=block_tables,
            cache_k_dequant_scales=cache_k_dequant_scales, 
            cache_k_quant_scales=cache_k_quant_scales, 
            cache_v_dequant_scales=cache_v_dequant_scales, 
            cache_v_quant_scales=cache_v_quant_scales,            
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_q=cu_seqlens_q,
            cum_offsets=cum_offsets,
            key_cache=key_cache,
            mask=att_mask,
            out_shift=out_shift,
            out_smooth=out_smooth,
            padding_offsets=padding_offsets,
            qkv=qkv_matmul,
            qkv_out_scale=qkv_out_scale,
            rope_emb=cos_table,
            seq_lens_decoder=seq_lens_decoder,
            seq_lens_encoder=seq_lens_encoder,
            seq_lens_this_time=seq_lens_this_time,
            tgt_mask=sin_table,
            value_cache=value_cache)

        allreduce_sum1 = dequant_linear(attention_.Output("fmha_out"), proj_weight, out_scale)
        ffn_rms_norm = ir.PassDesc.OP.rms_norm(norm_weight=ffn_in_scale, residual=lookup, x=allreduce_sum1)
        ffn_matmul1 = ir.PassDesc.OP.matmul_v2(X=ffn_rms_norm.Output("out"), Y=ffn1_weight)
        fused_bias_act_ = ir.PassDesc.OP.fused_bias_act(dequant_scales=ffn1_out_scale, shift=ffn2_shift, smooth=ffn2_smooth, x=ffn_matmul1)
        allreduce_sum2 = dequant_linear(fused_bias_act_, ffn2_weight, ffn2_out_scale)

        return ffn_rms_norm.Output("residual_out")[0], allreduce_sum2

    def replace(lookup, in_scale, qkv_weight, proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight, block_tables, cu_seqlens_k, cu_seqlens_q, 
            cum_offsets, key_cache, out_shift, out_smooth, padding_offsets, qkv_out_scale, seq_lens_decoder, seq_lens_encoder, seq_lens_this_time, 
            value_cache, out_scale, ffn1_out_scale, ffn2_shift, ffn2_smooth, ffn2_out_scale, att_mask, cos_table, sin_table,
            cache_k_dequant_scales, cache_k_quant_scales, cache_v_dequant_scales, cache_v_quant_scales):
        result = llama_paralle_cached_layer_adaptor_65B(lookup, in_scale, qkv_weight, qkv_out_scale, proj_weight, out_shift, out_smooth, out_scale, ffn_in_scale, ffn1_weight, ffn1_out_scale, ffn2_weight, ffn2_shift, ffn2_smooth, ffn2_out_scale, cos_table, sin_table,
                                                    att_mask, key_cache, value_cache, seq_lens_decoder, seq_lens_encoder, block_tables, cache_k_dequant_scales, cache_k_quant_scales, cache_v_dequant_scales, cache_v_quant_scales)

        return result[2], result[0]
        # add1 = ir.PassDesc.OP.my_add_n(X=lookup, Y=cu_seqlens_k, Z=cu_seqlens_q)
        # add2 = ir.PassDesc.OP.my_add_n(X=lookup, Y=add1, Z=cu_seqlens_q)
        # return add1, add2

    return pattern, replace

@ir.RegisterPass
def llama_fuse_attention_dynamic_parallel_others_65B():
    def pattern(in_scale, rms_norm_residual, matmul_, qkv_weight, proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight, block_tables, cu_seqlens_k, cu_seqlens_q, 
            cum_offsets, key_cache, out_shift, out_smooth, padding_offsets, qkv_out_scale, seq_lens_decoder, seq_lens_encoder, seq_lens_this_time, 
            value_cache, out_scale, ffn1_out_scale, ffn2_shift, ffn2_smooth, ffn2_out_scale, att_mask, cos_table, sin_table, 
            cache_k_dequant_scales, cache_k_quant_scales, cache_v_dequant_scales, cache_v_quant_scales):
        rms_norm_ = ir.PassDesc.OP.rms_norm(norm_weight=in_scale, residual=rms_norm_residual, x=matmul_)
        qkv_matmul = ir.PassDesc.OP.matmul_v2(X=rms_norm_.Output("out"), Y=qkv_weight)
        attention_ = ir.PassDesc.OP.block_multihead_attention(
            block_tables=block_tables,
            cache_k_dequant_scales=cache_k_dequant_scales, 
            cache_k_quant_scales=cache_k_quant_scales, 
            cache_v_dequant_scales=cache_v_dequant_scales, 
            cache_v_quant_scales=cache_v_quant_scales,               
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_q=cu_seqlens_q,
            cum_offsets=cum_offsets,
            key_cache=key_cache,
            mask=att_mask,
            out_shift=out_shift,
            out_smooth=out_smooth,
            padding_offsets=padding_offsets,
            qkv=qkv_matmul,
            qkv_out_scale=qkv_out_scale,
            rope_emb=cos_table,
            seq_lens_decoder=seq_lens_decoder,
            seq_lens_encoder=seq_lens_encoder,
            seq_lens_this_time=seq_lens_this_time,
            tgt_mask=sin_table,
            value_cache=value_cache)

        allreduce_sum1 = dequant_linear(attention_.Output("fmha_out"), proj_weight, out_scale)
        ffn_rms_norm = ir.PassDesc.OP.rms_norm(norm_weight=ffn_in_scale, residual=rms_norm_.Output("residual_out"), x=allreduce_sum1)
        ffn_matmul1 = ir.PassDesc.OP.matmul_v2(X=ffn_rms_norm.Output("out"), Y=ffn1_weight)
        fused_bias_act_ = ir.PassDesc.OP.fused_bias_act(dequant_scales=ffn1_out_scale, shift=ffn2_shift, smooth=ffn2_smooth, x=ffn_matmul1)
        allreduce_sum2 = dequant_linear(fused_bias_act_, ffn2_weight, ffn2_out_scale)

        return ffn_rms_norm.Output("residual_out")[0], allreduce_sum2

    def replace(in_scale, rms_norm_residual, matmul_, qkv_weight, proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight, block_tables, cu_seqlens_k, cu_seqlens_q, 
            cum_offsets, key_cache, out_shift, out_smooth, padding_offsets, qkv_out_scale, seq_lens_decoder, seq_lens_encoder, seq_lens_this_time, 
            value_cache, out_scale, ffn1_out_scale, ffn2_shift, ffn2_smooth, ffn2_out_scale, att_mask, cos_table, sin_table,
            cache_k_dequant_scales, cache_k_quant_scales, cache_v_dequant_scales, cache_v_quant_scales):
        result = llama_paralle_cached_layer_adaptor_65B(matmul_, in_scale, qkv_weight, qkv_out_scale, proj_weight, out_shift, out_smooth, out_scale, ffn_in_scale, ffn1_weight, ffn1_out_scale, ffn2_weight, ffn2_shift, ffn2_smooth, ffn2_out_scale, cos_table, sin_table,
                                                    att_mask, key_cache, value_cache, seq_lens_decoder, seq_lens_encoder, block_tables, cache_k_dequant_scales, cache_k_quant_scales, cache_v_dequant_scales, cache_v_quant_scales)
        return result[2], result[0]
        # add1 = ir.PassDesc.OP.my_add_n(X=matmul_, Y=rms_norm_residual, Z=cu_seqlens_q)
        # add2 = ir.PassDesc.OP.my_add_n(X=matmul_, Y=add1, Z=cu_seqlens_q)        
        # return add1, add2

    return pattern, replace


def llama_paralle_smooth_cached_layer_adaptor_65B(lookup, in_scale, qkv_weight, qkv_out_scale, proj_weight, out_shift, out_smooth, out_scale, ffn_in_scale, ffn1_weight, ffn1_out_scale, ffn2_weight, ffn2_shift, ffn2_smooth, ffn2_out_scale, cos_table, sin_table,
                                       att_mask, key_cache, value_cache, seq_lens_decoder, seq_lens_encoder, block_tables):
    llama_decoder_layer_parallel_op = ir.PassDesc.OP.llama_blockattn_smooth_layer_parallel
    llama_decoder_layer_parallel_op._outputs = {}
    llama_decoder_layer_parallel_op(
           Hidden=lookup,
            NormWeight=in_scale,
            QKVMixWeight=qkv_weight,
            QKVDeqScale=qkv_out_scale,
            SelfOutLinearWeight=proj_weight,
            SelfOutLinearShift=out_shift,
            SelfOutLinearSmooth=out_smooth,
            SelfOutLinearDeqScale=out_scale,
            SelfOutNormWeight=ffn_in_scale,
            MlpGateUpWeight=ffn1_weight,
            MlpDeqScale=ffn1_out_scale,
            MlpDownWeight=ffn2_weight,
            MlpDownShift=ffn2_shift,
            MlpDownSmooth=ffn2_smooth,
            MlpDownDeqScale=ffn2_out_scale,
            CosTable=cos_table,
            SinTable=sin_table,
            AttentionMask=att_mask,
            Cache_K=key_cache,
            Cache_V=value_cache,
            DecoderSeqLength=seq_lens_decoder,
            EncoderSeqLength=seq_lens_encoder,
            BlockTables=block_tables,
    )

    outs_name = [paddle.base.unique_name.generate('llama_decoder_layer_parallel') for i in range(3)] # 3 outputs
    print(outs_name)
    llama_decoder_layer_parallel_op._desc.set_output("Out", [outs_name[0]])

    llama_decoder_layer_parallel_op.Attr("block_size").MappedPattern(op="block_multihead_attention", name="block_size", index=0)
    llama_decoder_layer_parallel_op.Attr("rmsNormEps").MappedPattern(op="rms_norm", name="epsilon", index=0)
    llama_decoder_layer_parallel_op.Attr("inputRmsNormScale").MappedPattern(op="rms_norm", name="quant_scale", index=0)
    llama_decoder_layer_parallel_op.Attr("selfRmsNormScale").MappedPattern(op="rms_norm", name="quant_scale", index=1)
    llama_decoder_layer_parallel_op.Attr("selfQuantScale").MappedPattern(op="block_multihead_attention", name="out_scale", index=0)
    llama_decoder_layer_parallel_op.Attr("mlpQuantScale").MappedPattern(op="fused_bias_act", name="quant_scale", index=0)

    block = paddle.static.default_main_program().current_block()
    results = []
    for out in outs_name:
        results.append(block.create_var(name=out))
    return results[0], results[1], results[2]

@ir.RegisterPass
def llama_fuse_attention_dynamic_smooth_parallel_first_65B():
    def pattern(lookup, in_scale, qkv_weight, proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight, block_tables, cu_seqlens_k, cu_seqlens_q, 
            cum_offsets, key_cache, out_shift, out_smooth, padding_offsets, qkv_out_scale, seq_lens_decoder, seq_lens_encoder, seq_lens_this_time, 
            value_cache, out_scale, ffn1_out_scale, ffn2_shift, ffn2_smooth, ffn2_out_scale, att_mask, cos_table, sin_table):
        rms_norm_ = ir.PassDesc.OP.rms_norm(norm_weight=in_scale, x=lookup)
        qkv_matmul = ir.PassDesc.OP.matmul_v2(X=rms_norm_.Output("out"), Y=qkv_weight)
        attention_ = ir.PassDesc.OP.block_multihead_attention(
            block_tables=block_tables,
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_q=cu_seqlens_q,
            cum_offsets=cum_offsets,
            key_cache=key_cache,
            mask=att_mask,
            out_shift=out_shift,
            out_smooth=out_smooth,
            padding_offsets=padding_offsets,
            qkv=qkv_matmul,
            qkv_out_scale=qkv_out_scale,
            rope_emb=cos_table,
            seq_lens_decoder=seq_lens_decoder,
            seq_lens_encoder=seq_lens_encoder,
            seq_lens_this_time=seq_lens_this_time,
            tgt_mask=sin_table,
            value_cache=value_cache)

        allreduce_sum1 = dequant_linear(attention_.Output("fmha_out"), proj_weight, out_scale)
        ffn_rms_norm = ir.PassDesc.OP.rms_norm(norm_weight=ffn_in_scale, residual=lookup, x=allreduce_sum1)
        ffn_matmul1 = ir.PassDesc.OP.matmul_v2(X=ffn_rms_norm.Output("out"), Y=ffn1_weight)
        fused_bias_act_ = ir.PassDesc.OP.fused_bias_act(dequant_scales=ffn1_out_scale, shift=ffn2_shift, smooth=ffn2_smooth, x=ffn_matmul1)
        allreduce_sum2 = dequant_linear(fused_bias_act_, ffn2_weight, ffn2_out_scale)

        return ffn_rms_norm.Output("residual_out")[0], allreduce_sum2

    def replace(lookup, in_scale, qkv_weight, proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight, block_tables, cu_seqlens_k, cu_seqlens_q, 
            cum_offsets, key_cache, out_shift, out_smooth, padding_offsets, qkv_out_scale, seq_lens_decoder, seq_lens_encoder, seq_lens_this_time, 
            value_cache, out_scale, ffn1_out_scale, ffn2_shift, ffn2_smooth, ffn2_out_scale, att_mask, cos_table, sin_table):
        result = llama_paralle_smooth_cached_layer_adaptor_65B(lookup, in_scale, qkv_weight, qkv_out_scale, proj_weight, out_shift, out_smooth, out_scale, ffn_in_scale, ffn1_weight, ffn1_out_scale, ffn2_weight, ffn2_shift, ffn2_smooth, ffn2_out_scale, cos_table, sin_table,
                                                    att_mask, key_cache, value_cache, seq_lens_decoder, seq_lens_encoder, block_tables)

        return result[2], result[0]

    return pattern, replace

@ir.RegisterPass
def llama_fuse_attention_dynamic_smooth_parallel_others_65B():
    def pattern(in_scale, rms_norm_residual, matmul_, qkv_weight, proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight, block_tables, cu_seqlens_k, cu_seqlens_q, 
            cum_offsets, key_cache, out_shift, out_smooth, padding_offsets, qkv_out_scale, seq_lens_decoder, seq_lens_encoder, seq_lens_this_time, 
            value_cache, out_scale, ffn1_out_scale, ffn2_shift, ffn2_smooth, ffn2_out_scale, att_mask, cos_table, sin_table):
        rms_norm_ = ir.PassDesc.OP.rms_norm(norm_weight=in_scale, residual=rms_norm_residual, x=matmul_)
        qkv_matmul = ir.PassDesc.OP.matmul_v2(X=rms_norm_.Output("out"), Y=qkv_weight)
        attention_ = ir.PassDesc.OP.block_multihead_attention(
            block_tables=block_tables,
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_q=cu_seqlens_q,
            cum_offsets=cum_offsets,
            key_cache=key_cache,
            mask=att_mask,
            out_shift=out_shift,
            out_smooth=out_smooth,
            padding_offsets=padding_offsets,
            qkv=qkv_matmul,
            qkv_out_scale=qkv_out_scale,
            rope_emb=cos_table,
            seq_lens_decoder=seq_lens_decoder,
            seq_lens_encoder=seq_lens_encoder,
            seq_lens_this_time=seq_lens_this_time,
            tgt_mask=sin_table,
            value_cache=value_cache)

        allreduce_sum1 = dequant_linear(attention_.Output("fmha_out"), proj_weight, out_scale)
        ffn_rms_norm = ir.PassDesc.OP.rms_norm(norm_weight=ffn_in_scale, residual=rms_norm_.Output("residual_out"), x=allreduce_sum1)
        ffn_matmul1 = ir.PassDesc.OP.matmul_v2(X=ffn_rms_norm.Output("out"), Y=ffn1_weight)
        fused_bias_act_ = ir.PassDesc.OP.fused_bias_act(dequant_scales=ffn1_out_scale, shift=ffn2_shift, smooth=ffn2_smooth, x=ffn_matmul1)
        allreduce_sum2 = dequant_linear(fused_bias_act_, ffn2_weight, ffn2_out_scale)

        return ffn_rms_norm.Output("residual_out")[0], allreduce_sum2

    def replace(in_scale, rms_norm_residual, matmul_, qkv_weight, proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight, block_tables, cu_seqlens_k, cu_seqlens_q, 
            cum_offsets, key_cache, out_shift, out_smooth, padding_offsets, qkv_out_scale, seq_lens_decoder, seq_lens_encoder, seq_lens_this_time, 
            value_cache, out_scale, ffn1_out_scale, ffn2_shift, ffn2_smooth, ffn2_out_scale, att_mask, cos_table, sin_table):
        result = llama_paralle_smooth_cached_layer_adaptor_65B(matmul_, in_scale, qkv_weight, qkv_out_scale, proj_weight, out_shift, out_smooth, out_scale, ffn_in_scale, ffn1_weight, ffn1_out_scale, ffn2_weight, ffn2_shift, ffn2_smooth, ffn2_out_scale, cos_table, sin_table,
                                                    att_mask, key_cache, value_cache, seq_lens_decoder, seq_lens_encoder, block_tables)
        return result[2], result[0]

    return pattern, replace