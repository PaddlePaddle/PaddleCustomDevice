import paddle
from paddle.incubate.passes import ir

def transpose_without_axis(x):
    op = ir.PassDesc.OP.transpose2
    op._outputs.pop("XShape")
    return op(X=x)

def reshape_without_shape(x, shape_tensor):
    op = ir.PassDesc.OP.reshape2
    op._outputs.pop("XShape")
    return op(X=x, ShapeTensor=shape_tensor)

def reshape_with_shape(x, shape):
    op = ir.PassDesc.OP.reshape2
    # op.SetAttr("shape", shape)
    op._outputs.pop("XShape")
    return op(X=x)

def concat_in_axis_1(x, y):
    op = ir.PassDesc.OP.concat
    op.SetAttr("axis", 1)
    return op(X=[x, y])

def rmsnorm_output(x, y):
    cast_tmp1 = ir.PassDesc.OP.cast(X=x)
    cast_tmp2 = ir.PassDesc.OP.cast(X=x)
    pow_tmp = ir.PassDesc.OP.pow(X=cast_tmp1)
    reduce_mean_tmp = ir.PassDesc.OP.reduce_mean(X=pow_tmp)
    scale_tmp = ir.PassDesc.OP.scale(X=reduce_mean_tmp)
    rsqrt_tmp = ir.PassDesc.OP.rsqrt(X=scale_tmp)
    mul = ir.PassDesc.OP.elementwise_mul(X=rsqrt_tmp, Y=cast_tmp2)
    cast_tmp = ir.PassDesc.OP.cast(X=mul)
    return ir.PassDesc.OP.elementwise_mul(X=cast_tmp, Y=y)

def mlp_output(x, y_linear1, y_linear2, y_linear3):
    matmul_linear1 = ir.PassDesc.OP.matmul_v2(X=x, Y=y_linear1)
    silu_tmp = ir.PassDesc.OP.silu(X=matmul_linear1)
    matmul_linear2 = ir.PassDesc.OP.matmul_v2(X=x, Y=y_linear2)
    mul = ir.PassDesc.OP.elementwise_mul(X=silu_tmp, Y=matmul_linear2)
    return ir.PassDesc.OP.matmul_v2(X=mul, Y=y_linear3)

def mlp_multi_output(x, y_linear1, y_linear2, y_linear3):
    identity1 = ir.PassDesc.OP.c_identity(X=x)
    matmul_linear1 = ir.PassDesc.OP.matmul_v2(X=identity1, Y=y_linear1)
    silu_tmp = ir.PassDesc.OP.silu(X=matmul_linear1)
    identity2 = ir.PassDesc.OP.c_identity(X=x)
    matmul_linear2 = ir.PassDesc.OP.matmul_v2(X=identity2, Y=y_linear2)       
    mul = ir.PassDesc.OP.elementwise_mul(X=silu_tmp, Y=matmul_linear2)
    return ir.PassDesc.OP.matmul_v2(X=mul, Y=y_linear3) 

def cal_sin_cos(reshape_k, eager, id_input):
    slice_ = ir.PassDesc.OP.slice(Input=eager, EndsTensorList=reshape_k)

    cast_op = ir.PassDesc.OP.cast
    cast_op.SetAttr("in_dtype", 5)
    cast_op.SetAttr("out_dtype", 4)
    cast_ = cast_op(X=slice_)

    squeeze_op = ir.PassDesc.OP.squeeze2
    squeeze_op._outputs.pop("XShape")
    squeeze_ = squeeze_op(X=cast_)

    transpose2_op = ir.PassDesc.OP.transpose2
    transpose2_op.SetAttr("axis", [0, 1])
    transpose2_op._outputs.pop("XShape")
    transpose_ = transpose2_op(X=squeeze_)

    stack_ = ir.PassDesc.OP.stack(X=id_input)

    gather_nd_ = ir.PassDesc.OP.gather_nd(X=transpose_, Index=stack_)
    unsqueeze_op = ir.PassDesc.OP.unsqueeze2
    unsqueeze_op._outputs.pop("XShape")
    return unsqueeze_op(X=gather_nd_)

def cal_rope(reshape, sin, cos):
    # rotate_half
    slice1 = ir.PassDesc.OP.slice(Input=reshape)

    slice2 = ir.PassDesc.OP.slice(Input=reshape)
    
    scale_ = ir.PassDesc.OP.scale(X=slice2)

    concat_op = ir.PassDesc.OP.concat
    concat_op.SetAttr("axis", -1)
    concat_ = concat_op(X=[scale_, slice1])

    # (q * cos) + (rotate_half(q) * sin)
    mul1 = ir.PassDesc.OP.elementwise_mul(X=reshape, Y=cos)
    mul2 = ir.PassDesc.OP.elementwise_mul(X=concat_, Y=sin)

    return ir.PassDesc.OP.elementwise_add(X=mul1, Y=mul2)

def generate_llama_attention(embedded_q, embedded_k, embedded_v, linear_weight, lookup, reshape_v_tmp):
    transed_q = transpose_without_axis(embedded_q)
    scaled_q = ir.PassDesc.OP.scale(X=transed_q)

    transed_k = transpose_without_axis(embedded_k)
    double_transed_k = transpose_without_axis(transed_k)

    q_mul_k = ir.PassDesc.OP.matmul_v2(X=scaled_q, Y=double_transed_k)

    shape_q =  ir.PassDesc.OP.shape(Input=embedded_q)
    slice0_q =  ir.PassDesc.OP.slice(Input=shape_q)
    slice1_q =  ir.PassDesc.OP.slice(Input=shape_q)

    shape_v =  ir.PassDesc.OP.shape(Input=embedded_v)
    slice_v =  ir.PassDesc.OP.slice(Input=shape_v)

    reshape_v_const1 = ir.PassDesc.OP.fill_constant()
    reshape_v = reshape_without_shape(reshape_v_tmp, [slice0_q, reshape_v_const, slice1_q, slice_v])


    added_attn_weight = ir.PassDesc.OP.elementwise_add(X=q_mul_k, Y=reshape_v)
    casted_attn_weight = ir.PassDesc.OP.cast(X=added_attn_weight)
    
    softamxed_attn_weight = ir.PassDesc.OP.softmax(X=casted_attn_weight)
    casted_softamxed = ir.PassDesc.OP.cast(X=softamxed_attn_weight)

    transed_v = transpose_without_axis(embedded_v)

    out = ir.PassDesc.OP.matmul_v2(X=casted_softamxed, Y=transed_v)
    out = transpose_without_axis(out)

    reshape_const1 = ir.PassDesc.OP.fill_constant()
    reshape_out = reshape_without_shape(out, [slice0_q, slice1_q, reshape_const1])

    matmuled_reshape_out = ir.PassDesc.OP.matmul_v2(X=reshape_out, Y=linear_weight)
    
    added_result = ir.PassDesc.OP.elementwise_add(X=lookup, Y=matmuled_reshape_out)

    return added_result

def generate_llama_attention_parallel(embedded_q, embedded_k, embedded_v, linear_weight, lookup, reshape_v_tmp):
    transed_q = transpose_without_axis(embedded_q)
    scaled_q = ir.PassDesc.OP.scale(X=transed_q)

    transed_k = transpose_without_axis(embedded_k)
    double_transed_k = transpose_without_axis(transed_k)

    q_mul_k = ir.PassDesc.OP.matmul_v2(X=scaled_q, Y=double_transed_k)

    shape_q =  ir.PassDesc.OP.shape(Input=embedded_q)
    slice0_q =  ir.PassDesc.OP.slice(Input=shape_q)
    slice1_q =  ir.PassDesc.OP.slice(Input=shape_q)

    shape_v =  ir.PassDesc.OP.shape(Input=embedded_v)
    slice_v =  ir.PassDesc.OP.slice(Input=shape_v)


    reshape_v = reshape_without_shape(reshape_v_tmp, [slice0_q, slice1_q, slice_v])


    added_attn_weight = ir.PassDesc.OP.elementwise_add(X=q_mul_k, Y=reshape_v)
    casted_attn_weight = ir.PassDesc.OP.cast(X=added_attn_weight)
    
    softamxed_attn_weight = ir.PassDesc.OP.softmax(X=casted_attn_weight)
    casted_softamxed = ir.PassDesc.OP.cast(X=softamxed_attn_weight)

    transed_v = transpose_without_axis(embedded_v)

    out = ir.PassDesc.OP.matmul_v2(X=casted_softamxed, Y=transed_v)
    out = transpose_without_axis(out)

    reshape_out = reshape_without_shape(out, [slice0_q, slice1_q])

    matmuled_reshape_out = ir.PassDesc.OP.matmul_v2(X=reshape_out, Y=linear_weight)
    
    allreduce_sum = ir.PassDesc.OP.mp_allreduce_sum(X=matmuled_reshape_out)

    added_result = ir.PassDesc.OP.elementwise_add(X=lookup, Y=allreduce_sum)

    return added_result
    

def generate_llama_attention_cache_parallel(embedded_q, embedded_k, embedded_v, linear_weight, lookup, reshape_v_tmp):
    transed_q = transpose_without_axis(embedded_q)
    scaled_q = ir.PassDesc.OP.scale(X=transed_q)

    transed_k = transpose_without_axis(embedded_k)
    double_transed_k = transpose_without_axis(transed_k)

    q_mul_k = ir.PassDesc.OP.matmul_v2(X=scaled_q, Y=double_transed_k)

    shape_q =  ir.PassDesc.OP.shape(Input=embedded_q)
    slice0_q =  ir.PassDesc.OP.slice(Input=shape_q)

    shape_v =  ir.PassDesc.OP.shape(Input=embedded_v)
    slice_v =  ir.PassDesc.OP.slice(Input=shape_v)

    reshape_v = reshape_without_shape(reshape_v_tmp, [slice0_q, slice_v])


    added_attn_weight = ir.PassDesc.OP.elementwise_add(X=q_mul_k, Y=reshape_v)
    casted_attn_weight = ir.PassDesc.OP.cast(X=added_attn_weight)
    
    softamxed_attn_weight = ir.PassDesc.OP.softmax(X=casted_attn_weight)
    casted_softamxed = ir.PassDesc.OP.cast(X=softamxed_attn_weight)

    transed_v = transpose_without_axis(embedded_v)

    out = ir.PassDesc.OP.matmul_v2(X=casted_softamxed, Y=transed_v)
    out = transpose_without_axis(out)

    reshape_out = reshape_without_shape(out, [slice0_q])

    matmuled_reshape_out = ir.PassDesc.OP.matmul_v2(X=reshape_out, Y=linear_weight)
    
    allreduce_sum = ir.PassDesc.OP.mp_allreduce_sum(X=matmuled_reshape_out)

    added_result = ir.PassDesc.OP.elementwise_add(X=lookup, Y=allreduce_sum)

    return added_result

def fill_constant_with_value(value):
    op = ir.PassDesc.OP.fill_constant
    op.SetAttr("dtype", 2)
    op.SetAttr("shape", [1])
    op.SetAttr("value", value)
    return op()

@ir.RegisterPass
def llama_fuse_attention_layer():
    def pattern(lookup, norm_weight, q_weight, k_weight, v_weight, sin, cos, id_input, selfout_linear_weight, selfout_norm_weight, mlp_gate_weight, mlp_up_weight, mlp_down_weight, reshape_v_tmp):
        rmsnorm = rmsnorm_output(lookup, norm_weight)

        q_matmul = ir.PassDesc.OP.matmul_v2(X=rmsnorm, Y=q_weight)
        k_matmul = ir.PassDesc.OP.matmul_v2(X=rmsnorm, Y=k_weight)
        v_matmul = ir.PassDesc.OP.matmul_v2(X=rmsnorm, Y=v_weight)

        reshape_q = reshape_with_shape(q_matmul, [0, 0, 32, 128])
        reshape_k = reshape_with_shape(k_matmul, [0, 0, 32, 128])
        reshape_v = reshape_with_shape(v_matmul, [0, 0, 32, 128])

        # cal sin&&cos
        shape_ = ir.PassDesc.OP.shape(Input=reshape_k)
        slice_ = ir.PassDesc.OP.slice(Input=shape_)

        sin_ = cal_sin_cos(slice_, sin, id_input)
        cos_ = cal_sin_cos(slice_, cos, id_input)
        
        emb_q = cal_rope(reshape_q, sin_, cos_)
        emb_k = cal_rope(reshape_k, sin_, cos_)
        attention_ = generate_llama_attention(emb_q, emb_k, reshape_v, selfout_linear_weight, lookup, reshape_v_tmp)

        rmsnorm_mul = rmsnorm_output(attention_, selfout_norm_weight)
        mlp = mlp_output(rmsnorm_mul, mlp_gate_weight, mlp_up_weight, mlp_down_weight)
        return ir.PassDesc.OP.elementwise_add(X=attention_, Y=mlp), emb_k, reshape_v

    def replace(lookup, norm_weight, q_weight, k_weight, v_weight, sin, cos, id_input, selfout_linear_weight, selfout_norm_weight, mlp_gate_weight, mlp_up_weight, mlp_down_weight, reshape_v_tmp):
        # stub 
        add = ir.PassDesc.OP.my_add_n(X=lookup, Y=id_input, Z=reshape_v_tmp)
        return add, add, add
    return pattern, replace

@ir.RegisterPass
def llama_fuse_attention_cached_layer():
    def pattern(lookup, norm_weight, q_weight, k_weight, v_weight, sin, cos, id_input, selfout_linear_weight, selfout_norm_weight, mlp_gate_weight, mlp_up_weight, mlp_down_weight, reshape_v_tmp, cache_k, cache_v):
        rmsnorm = rmsnorm_output(lookup, norm_weight)

        q_matmul = ir.PassDesc.OP.matmul_v2(X=rmsnorm, Y=q_weight)
        k_matmul = ir.PassDesc.OP.matmul_v2(X=rmsnorm, Y=k_weight)
        v_matmul = ir.PassDesc.OP.matmul_v2(X=rmsnorm, Y=v_weight)

        reshape_q = reshape_with_shape(q_matmul, [0, 0, 32, 128])
        reshape_k = reshape_with_shape(k_matmul, [0, 0, 32, 128])
        reshape_v = reshape_with_shape(v_matmul, [0, 0, 32, 128])

        # cal sin&&cos
        shape_ = ir.PassDesc.OP.shape(Input=reshape_k)
        slice_ = ir.PassDesc.OP.slice(Input=shape_)

        sin_ = x(slice_, sin, id_input)
        cos_ = cal_sin_cos(slice_, cos, id_input)
        
        emb_q = cal_rope(reshape_q, sin_, cos_)
        emb_k = cal_rope(reshape_k, sin_, cos_)

        concated_k = concat_in_axis_1(cache_k, emb_k)
        concated_v = concat_in_axis_1(cache_v, reshape_v)

        attention_ = generate_llama_attention(emb_q, concated_k, concated_v, selfout_linear_weight, lookup, reshape_v_tmp)

        rmsnorm_mul = rmsnorm_output(attention_, selfout_norm_weight)
        mlp = mlp_output(rmsnorm_mul, mlp_gate_weight, mlp_up_weight, mlp_down_weight)
        return ir.PassDesc.OP.elementwise_add(X=attention_, Y=mlp), concated_k, concated_v

    def replace(lookup, norm_weight, q_weight, k_weight, v_weight, sin, cos, id_input, selfout_linear_weight, selfout_norm_weight, mlp_gate_weight, mlp_up_weight, mlp_down_weight, reshape_v_tmp, cache_k, cache_v):
        # stub 
        add = ir.PassDesc.OP.my_add_n(X=lookup, Y=id_input, Z=reshape_v_tmp)
        return add, add, add
    return pattern, replace

@ir.RegisterPass
def llama_fuse_attention_parallel_layer():
    def pattern(lookup, norm_weight, q_weight, k_weight, v_weight, selfout_linear_weight, selfout_norm_weight, mlp_gate_weight, mlp_up_weight, mlp_down_weight, reshape_v_tmp, sin, cos, id_input):
        rmsnorm = rmsnorm_output(lookup, norm_weight)

        identity1 = ir.PassDesc.OP.c_identity(X=rmsnorm)
        identity2 = ir.PassDesc.OP.c_identity(X=rmsnorm)
        identity3 = ir.PassDesc.OP.c_identity(X=rmsnorm)

        q_matmul = ir.PassDesc.OP.matmul_v2(X=identity1, Y=q_weight)
        k_matmul = ir.PassDesc.OP.matmul_v2(X=identity2, Y=k_weight)
        v_matmul = ir.PassDesc.OP.matmul_v2(X=identity3, Y=v_weight)

        reshape_q = reshape_with_shape(q_matmul, [0, 0, 4, 128])
        reshape_k = reshape_with_shape(k_matmul, [0, 0, 4, 128])
        reshape_v = reshape_with_shape(v_matmul, [0, 0, 4, 128])

        # cal sin&&cos
        shape_ = ir.PassDesc.OP.shape(Input=reshape_k)
        slice_ = ir.PassDesc.OP.slice(Input=shape_)

        sin_ = cal_sin_cos(slice_, sin, id_input)
        cos_ = cal_sin_cos(slice_, cos, id_input)
        emb_q = cal_rope(reshape_q, sin_, cos_)
        emb_k = cal_rope(reshape_k, sin_, cos_)
        attention_ = generate_llama_attention_parallel(emb_q, emb_k, reshape_v, selfout_linear_weight, lookup, reshape_v_tmp)

        rmsnorm_mul = rmsnorm_output(attention_, selfout_norm_weight)
        mlp = mlp_multi_output(rmsnorm_mul, mlp_gate_weight, mlp_up_weight, mlp_down_weight)
        allreduce_sum = ir.PassDesc.OP.mp_allreduce_sum(X=mlp)
        return ir.PassDesc.OP.elementwise_add(X=attention_, Y=allreduce_sum), emb_k, reshape_v

    def llama_layer_adaptor(lookup, norm_weight, q_weight, k_weight, v_weight, selfout_linear_weight, selfout_norm_weight, mlp_gate_weight, mlp_up_weight, mlp_down_weight, reshape_v_tmp, sin, cos, id_input):
        llama_encoder_layer_parallel_op = ir.PassDesc.OP.llama_encoder_layer_parallel
        llama_encoder_layer_parallel_op._outputs = {}
        llama_encoder_layer_parallel_op(
            Hidden=lookup,
            NormWeight=norm_weight,
            QMixWeight=q_weight, 
            KMixWeight=k_weight, 
            VMixWeight=v_weight, 
            SelfOutLinearWeight=selfout_linear_weight, 
            SelfOutNormWeight=selfout_norm_weight, 
            MlpGateWeight=mlp_gate_weight, 
            MlpDownWeight=mlp_down_weight, 
            MlpUpWeight=mlp_up_weight, 
            PositionIDs=id_input, 
            CosTable=cos, 
            SinTable=sin, 
            AttentionMask=reshape_v_tmp
        )
        outs_name = [paddle.base.unique_name.generate('llama_encoder_layer_parallel') for i in range(3)] # 3 outputs
        print(outs_name)
        llama_encoder_layer_parallel_op._desc.set_output("Out", [outs_name[0]])
        llama_encoder_layer_parallel_op._desc.set_output("PresentKey", [outs_name[1]])
        llama_encoder_layer_parallel_op._desc.set_output("PresentValue", [outs_name[2]])

        llama_encoder_layer_parallel_op.Attr("rmsNormEps").MappedPattern(op="scale", name="bias", index=0)
        llama_encoder_layer_parallel_op.Attr("shape").MappedPattern(op="reshape2", name="shape", index=0) # [0, 0, head_num, head_dim]
        # llama_encoder_layer_parallel_op.Attr("dk").MappedPattern(op="reshape2", name="shape", index=0)
        
        block = paddle.static.default_main_program().current_block()
        results = []
        for out in outs_name:
            results.append(block.create_var(name=out))
        return results[0], results[1], results[2]

    def replace(lookup, norm_weight, q_weight, k_weight, v_weight, selfout_linear_weight, selfout_norm_weight, mlp_gate_weight, mlp_up_weight, mlp_down_weight, reshape_v_tmp, sin, cos, id_input):
        # stub 
        out = llama_layer_adaptor(lookup, norm_weight, q_weight, k_weight, v_weight, selfout_linear_weight, selfout_norm_weight, mlp_gate_weight, mlp_up_weight, mlp_down_weight, reshape_v_tmp, sin, cos, id_input)
        return out[0], out[1], out[2]
    return pattern, replace

@ir.RegisterPass
def llama_fuse_attention_cached_parallel_layer():
    def pattern(lookup, norm_weight, q_weight, k_weight, v_weight, selfout_linear_weight, selfout_norm_weight, mlp_gate_weight, mlp_up_weight, mlp_down_weight, reshape_v_tmp, sin, cos, id_input, cache_k, cache_v):
        rmsnorm = rmsnorm_output(lookup, norm_weight)

        identity1 = ir.PassDesc.OP.c_identity(X=rmsnorm)
        identity2 = ir.PassDesc.OP.c_identity(X=rmsnorm)
        identity3 = ir.PassDesc.OP.c_identity(X=rmsnorm)

        q_matmul = ir.PassDesc.OP.matmul_v2(X=identity1, Y=q_weight)
        k_matmul = ir.PassDesc.OP.matmul_v2(X=identity2, Y=k_weight)
        v_matmul = ir.PassDesc.OP.matmul_v2(X=identity3, Y=v_weight)

        reshape_q = reshape_with_shape(q_matmul, [0, 0, 4, 128])
        reshape_k = reshape_with_shape(k_matmul, [0, 0, 4, 128])
        reshape_v = reshape_with_shape(v_matmul, [0, 0, 4, 128])

        # cal sin&&cos
        shape_ = ir.PassDesc.OP.shape(Input=cache_k)
        slice_ = ir.PassDesc.OP.slice(Input=shape_)

        slice_scale = ir.PassDesc.OP.scale(X=slice_)

        sin_ = cal_sin_cos(slice_scale, sin, id_input)
        cos_ = cal_sin_cos(slice_scale, cos, id_input)
        emb_q = cal_rope(reshape_q, sin_, cos_)
        emb_k = cal_rope(reshape_k, sin_, cos_)

        concated_k = concat_in_axis_1(cache_k, emb_k)
        concated_v = concat_in_axis_1(cache_v, reshape_v)

        attention_ = generate_llama_attention_cache_parallel(emb_q, concated_k, concated_v, selfout_linear_weight, lookup, reshape_v_tmp)

        rmsnorm_mul = rmsnorm_output(attention_, selfout_norm_weight)
        mlp = mlp_multi_output(rmsnorm_mul, mlp_gate_weight, mlp_up_weight, mlp_down_weight)
        allreduce_sum = ir.PassDesc.OP.mp_allreduce_sum(X=mlp)

        assign_k = ir.PassDesc.OP.assign(X=concated_k)
        assign_v = ir.PassDesc.OP.assign(X=concated_v)
        return ir.PassDesc.OP.elementwise_add(X=attention_, Y=allreduce_sum), assign_k, assign_v,

    def llama_layer_cached_adaptor(lookup, norm_weight, q_weight, k_weight, v_weight, selfout_linear_weight, selfout_norm_weight, mlp_gate_weight, mlp_up_weight, mlp_down_weight, reshape_v_tmp, sin, cos, id_input, cache_k, cache_v):
        llama_decoder_layer_parallel_op = ir.PassDesc.OP.llama_decoder_layer_parallel
        llama_decoder_layer_parallel_op._outputs = {}
        llama_decoder_layer_parallel_op(
            Hidden=lookup,
            NormWeight=norm_weight,
            QMixWeight=q_weight, 
            KMixWeight=k_weight, 
            VMixWeight=v_weight, 
            SelfOutLinearWeight=selfout_linear_weight, 
            SelfOutNormWeight=selfout_norm_weight, 
            MlpGateWeight=mlp_gate_weight, 
            MlpDownWeight=mlp_down_weight, 
            MlpUpWeight=mlp_up_weight, 
            PositionIDs=id_input, 
            CosTable=cos, 
            SinTable=sin, 
            AttentionMask=reshape_v_tmp,
            CacheK=cache_k,
            CacheV=cache_v
        )
        outs_name = [paddle.base.unique_name.generate('llama_decoder_layer_parallel') for i in range(3)] # 3 outputs
        print(outs_name)
        llama_decoder_layer_parallel_op._desc.set_output("Out", [outs_name[0]])
        llama_decoder_layer_parallel_op._desc.set_output("PresentKey", [outs_name[1]])
        llama_decoder_layer_parallel_op._desc.set_output("PresentValue", [outs_name[2]])

        llama_decoder_layer_parallel_op.Attr("rmsNormEps").MappedPattern(op="scale", name="bias", index=0)
        # llama_decoder_layer_parallel_op.Attr("shape").MappedPattern(op="reshape2", name="shape", index=0) # [0, 0, head_num, head_dim]
        # llama_decoder_layer_parallel_op.Attr("headnum").MappedPattern(op="reshape2", name="shape", index=0) # [0, 0, head_num, head_dim]
        # llama_decoder_layer_parallel_op.Attr("dk").MappedPattern(op="reshape2", name="shape", index=0)
        
        block = paddle.static.default_main_program().current_block()
        results = []
        for out in outs_name:
            results.append(block.create_var(name=out))
        return results[0], results[1], results[2]

    def replace(lookup, norm_weight, q_weight, k_weight, v_weight, selfout_linear_weight, selfout_norm_weight, mlp_gate_weight, mlp_up_weight, mlp_down_weight, reshape_v_tmp, sin, cos, id_input, cache_k, cache_v):
        # stub 
        out = llama_layer_cached_adaptor(lookup, norm_weight, q_weight, k_weight, v_weight, selfout_linear_weight, selfout_norm_weight, mlp_gate_weight, mlp_up_weight, mlp_down_weight, reshape_v_tmp, sin, cos, id_input, cache_k, cache_v)
        return out[0], out[1], out[2]
    return pattern, replace

@ir.RegisterPass
def llama_fuse_attention_dynamic_layer1():
    def pattern(lookup, in_scale, qkv_weight, cache_kvs, rotary_t, sequence_l, mask, proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight):
        rms_norm_ = ir.PassDesc.OP.rms_norm(norm_weight=in_scale, x=lookup)
        qkv_matmul = ir.PassDesc.OP.matmul_v2(X=rms_norm_.Output("out"), Y=qkv_weight)
        attention_ = ir.PassDesc.OP.masked_multihead_attention(cache_kv=cache_kvs, rotary_tensor=rotary_t, sequence_lengths=sequence_l, src_mask=mask, x=qkv_matmul)
        proj_matmul = ir.PassDesc.OP.matmul_v2(X=attention_.Output("out"), Y=proj_weight)
        ffn_rms_norm = ir.PassDesc.OP.rms_norm(norm_weight=ffn_in_scale, residual=lookup, x=proj_matmul)
        ffn_matmul1 = ir.PassDesc.OP.matmul_v2(X=ffn_rms_norm.Output("out"), Y=ffn1_weight)        
        fused_bias_act_ = ir.PassDesc.OP.fused_bias_act(x=ffn_matmul1)
        ffn_matmul2 = ir.PassDesc.OP.matmul_v2(X=fused_bias_act_, Y=ffn2_weight)

        return ffn_rms_norm.Output("residual_out")[0], ffn_matmul2, attention_.Output("cache_kv_out")[0]
        
    def replace(lookup, in_scale, qkv_weight, cache_kvs, rotary_t, sequence_l, mask, proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight):
        # stub 
        result = ir.PassDesc.OP.llama_cached_layer(
            in_scale=in_scale, 
            rms_norm_residual=lookup, 
            matmul_=lookup, 
            qkv_weight=qkv_weight, 
            cache_kvs=cache_kvs, 
            rotary_t=rotary_t, 
            sequence_l=sequence_l, 
            mask=mask, 
            proj_weight=proj_weight, 
            ffn_in_scale=ffn_in_scale, 
            ffn1_weight=ffn1_weight, 
            ffn2_weight=ffn2_weight)
        return result.Output("norm_out")[0], result.Output("norm_residual")[0], result.Output("cached_kv")[0]

    return pattern, replace

@ir.RegisterPass
def llama_fuse_attention_dynamic_layer2():
    def pattern(in_scale, rms_norm_residual, matmul_, qkv_weight, cache_kvs, rotary_t, sequence_l, mask, proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight):
        rms_norm_ = ir.PassDesc.OP.rms_norm(norm_weight=in_scale, residual=rms_norm_residual, x=matmul_)
        qkv_matmul = ir.PassDesc.OP.matmul_v2(X=rms_norm_.Output("out"), Y=qkv_weight)
        attention_ = ir.PassDesc.OP.masked_multihead_attention(cache_kv=cache_kvs, rotary_tensor=rotary_t, sequence_lengths=sequence_l, src_mask=mask, x=qkv_matmul)
        proj_matmul = ir.PassDesc.OP.matmul_v2(X=attention_.Output("out"), Y=proj_weight)
        ffn_rms_norm = ir.PassDesc.OP.rms_norm(norm_weight=ffn_in_scale, residual=rms_norm_.Output("residual_out"), x=proj_matmul)
        ffn_matmul1 = ir.PassDesc.OP.matmul_v2(X=ffn_rms_norm.Output("out"), Y=ffn1_weight)        
        fused_bias_act_ = ir.PassDesc.OP.fused_bias_act(x=ffn_matmul1)
        ffn_matmul2 = ir.PassDesc.OP.matmul_v2(X=fused_bias_act_, Y=ffn2_weight)

        return ffn_rms_norm.Output("residual_out")[0], ffn_matmul2, attention_.Output("cache_kv_out")[0]
        
    def replace(in_scale, rms_norm_residual, matmul_, qkv_weight, cache_kvs, rotary_t, sequence_l, mask, proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight):
        # stub 
        result = ir.PassDesc.OP.llama_cached_layer(
            in_scale=in_scale, 
            rms_norm_residual=rms_norm_residual, 
            matmul_=matmul_, 
            qkv_weight=qkv_weight, 
            cache_kvs=cache_kvs, 
            rotary_t=rotary_t, 
            sequence_l=sequence_l, 
            mask=mask, 
            proj_weight=proj_weight, 
            ffn_in_scale=ffn_in_scale, 
            ffn1_weight=ffn1_weight, 
            ffn2_weight=ffn2_weight)
        return result.Output("norm_out")[0], result.Output("norm_residual")[0], result.Output("cached_kv")[0]

    return pattern, replace

# dynamic batch 8mp decoder
def llama_paralle_cached_layer_adaptor(lookup, in_scale, qkv_weight, cache_kvs, rotary_t, sequence_l, mask, proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight):
    llama_decoder_layer_parallel_op = ir.PassDesc.OP.llama_decoder_layer_parallel
    llama_decoder_layer_parallel_op._outputs = {}
    llama_decoder_layer_parallel_op(
            Hidden=lookup,
            NormWeight=in_scale,
            QKVMixWeight=qkv_weight,
            SelfOutLinearWeight=proj_weight,
            SelfOutNormWeight=ffn_in_scale,
            MlpGateUpWeight=ffn1_weight,
            MlpDownWeight=ffn2_weight,
            PositionIDs=sequence_l,
            CosSinTable=rotary_t,
            AttentionMask=mask,
            SeqLength=sequence_l,
            Cache_KV=cache_kvs,
    )

    outs_name = [paddle.base.unique_name.generate('llama_decoder_layer_parallel') for i in range(3)] # 3 outputs
    print(outs_name)
    llama_decoder_layer_parallel_op._desc.set_output("Out", [outs_name[0]])
    llama_decoder_layer_parallel_op._desc.set_output("PresentKV", [outs_name[1]])
    # llama_decoder_layer_parallel_op._desc.set_output("PresentValue", [outs_name[2]])

    llama_decoder_layer_parallel_op.Attr("rmsNormEps").MappedPattern(op="rms_norm", name="epsilon", index=0)
    # llama_decoder_layer_parallel_op.Attr("headDim").MappedPattern(op="qkv_transpose_split", name="head_size", index=0) # [0, 0, head_num, head_dim]
    # llama_decoder_layer_parallel_op.Attr("headNum").MappedPattern(op="qkv_transpose_split", name="num_head", index=0) # [0, 0, head_num, head_dim]
    # llama_encoder_layer_parallel_op.Attr("dk").MappedPattern(op="reshape2", name="shape", index=0)

    block = paddle.static.default_main_program().current_block()
    results = []
    for out in outs_name:
        results.append(block.create_var(name=out))
    return results[0], results[1], results[2]

@ir.RegisterPass
def llama_fuse_attention_dynamic_parallel_layer1():
    def pattern(lookup, in_scale, qkv_weight, cache_kvs, rotary_t, sequence_l, mask, proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight):
        rms_norm_ = ir.PassDesc.OP.rms_norm(norm_weight=in_scale, x=lookup)
        qkv_matmul = ir.PassDesc.OP.matmul_v2(X=rms_norm_.Output("out"), Y=qkv_weight)
        attention_ = ir.PassDesc.OP.masked_multihead_attention(cache_kv=cache_kvs, rotary_tensor=rotary_t, sequence_lengths=sequence_l, src_mask=mask, x=qkv_matmul)
        proj_matmul = ir.PassDesc.OP.matmul_v2(X=attention_.Output("out"), Y=proj_weight)
        allreduce_sum1 = ir.PassDesc.OP.c_allreduce_sum(X=proj_matmul)
        ffn_rms_norm = ir.PassDesc.OP.rms_norm(norm_weight=ffn_in_scale, residual=lookup, x=allreduce_sum1)
        ffn_matmul1 = ir.PassDesc.OP.matmul_v2(X=ffn_rms_norm.Output("out"), Y=ffn1_weight)        
        fused_bias_act_ = ir.PassDesc.OP.fused_bias_act(x=ffn_matmul1)
        ffn_matmul2 = ir.PassDesc.OP.matmul_v2(X=fused_bias_act_, Y=ffn2_weight)
        allreduce_sum2 = ir.PassDesc.OP.c_allreduce_sum(X=ffn_matmul2)

        return ffn_rms_norm.Output("residual_out")[0], allreduce_sum2, attention_.Output("cache_kv_out")[0]
        
    def replace(lookup, in_scale, qkv_weight, cache_kvs, rotary_t, sequence_l, mask, proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight):
        result = llama_paralle_cached_layer_adaptor(lookup, in_scale, qkv_weight, cache_kvs, rotary_t, sequence_l, mask, proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight)
        return result[2], result[0], result[1]

    return pattern, replace

@ir.RegisterPass
def llama_fuse_attention_dynamic_parallel_layer2():
    def pattern(in_scale, rms_norm_residual, matmul_, qkv_weight, cache_kvs, rotary_t, sequence_l, mask, proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight):
        rms_norm_ = ir.PassDesc.OP.rms_norm(norm_weight=in_scale, residual=rms_norm_residual, x=matmul_)
        qkv_matmul = ir.PassDesc.OP.matmul_v2(X=rms_norm_.Output("out"), Y=qkv_weight)
        attention_ = ir.PassDesc.OP.masked_multihead_attention(cache_kv=cache_kvs, rotary_tensor=rotary_t, sequence_lengths=sequence_l, src_mask=mask, x=qkv_matmul)
        proj_matmul = ir.PassDesc.OP.matmul_v2(X=attention_.Output("out"), Y=proj_weight)
        allreduce_sum1 = ir.PassDesc.OP.c_allreduce_sum(X=proj_matmul)        
        ffn_rms_norm = ir.PassDesc.OP.rms_norm(norm_weight=ffn_in_scale, residual=rms_norm_.Output("residual_out"), x=allreduce_sum1)
        ffn_matmul1 = ir.PassDesc.OP.matmul_v2(X=ffn_rms_norm.Output("out"), Y=ffn1_weight)        
        fused_bias_act_ = ir.PassDesc.OP.fused_bias_act(x=ffn_matmul1)
        ffn_matmul2 = ir.PassDesc.OP.matmul_v2(X=fused_bias_act_, Y=ffn2_weight)
        allreduce_sum2 = ir.PassDesc.OP.c_allreduce_sum(X=ffn_matmul2)        

        return ffn_rms_norm.Output("residual_out")[0], allreduce_sum2, attention_.Output("cache_kv_out")[0]
        
    def replace(in_scale, rms_norm_residual, matmul_, qkv_weight, cache_kvs, rotary_t, sequence_l, mask, proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight):
        result = llama_paralle_cached_layer_adaptor(matmul_, in_scale, qkv_weight, cache_kvs, rotary_t, sequence_l, mask, proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight)
        return result[2], result[0], result[1]

    return pattern, replace

# dynamic batch 8mp encoder
def llama_paralle_layer_adaptor(x, residual, input_ids, padding_offset, seq_len_encoder, cache_kv, mask, rotary_emb, ln_scale, qkv_weight, out_proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight):
    llama_encoder_layer_parallel_op = ir.PassDesc.OP.llama_encoder_layer_parallel
    llama_encoder_layer_parallel_op._outputs = {}
    llama_encoder_layer_parallel_op(
            Hidden=x,
            NormWeight=ln_scale,
            QKVMixWeight=qkv_weight,
            SelfOutLinearWeight=out_proj_weight,
            SelfOutNormWeight=ffn_in_scale,
            MlpGateUpWeight=ffn1_weight,
            MlpDownWeight=ffn2_weight,
            PositionIDs=input_ids,
            CosSinTable=rotary_emb,
            AttentionMask=mask,
            Cache_KV=cache_kv,
            SeqLength=seq_len_encoder,
    )
    outs_name = [paddle.base.unique_name.generate('llama_decoder_layer_parallel') for i in range(8)] # 3 outputs
    print(outs_name)
    llama_encoder_layer_parallel_op._desc.set_output("Out", [outs_name[0]])
    llama_encoder_layer_parallel_op._desc.set_output("PresentKV", [outs_name[1]])
    # llama_encoder_layer_parallel_op._desc.set_output("PresentValue", [outs_name[2]])

    llama_encoder_layer_parallel_op.Attr("rmsNormEps").MappedPattern(op="rms_norm", name="epsilon", index=0)
    llama_encoder_layer_parallel_op.Attr("headDim").MappedPattern(op="qkv_transpose_split", name="head_size", index=0) # [0, 0, head_num, head_dim]
    llama_encoder_layer_parallel_op.Attr("headNum").MappedPattern(op="qkv_transpose_split", name="num_head", index=0) # [0, 0, head_num, head_dim]
    # llama_encoder_layer_parallel_op.Attr("dk").MappedPattern(op="reshape2", name="shape", index=0)

    block = paddle.static.default_main_program().current_block()
    results = []
    for out in outs_name:
        results.append(block.create_var(name=out))
    return results[0], results[1], results[2], results[3], results[4], results[5],results[6], results[7]

@ir.RegisterPass
def llama_fuse_attention_dynamic_parallel_layer():
    def pattern(x, residual, input_ids, padding_offset, seq_len_encoder, cache_kv, mask, rotary_emb, ln_scale, qkv_weight, out_proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight):
        rms_norm_0 = ir.PassDesc.OP.rms_norm(norm_weight=ln_scale, residual=residual, x=x)
        qkv = ir.PassDesc.OP.matmul_v2(X=rms_norm_0.Output("out"), Y=qkv_weight)
        qkv_split = ir.PassDesc.OP.qkv_transpose_split(input_ids=input_ids, padding_offset=padding_offset, qkv=qkv, seq_lens=seq_len_encoder)
        q = qkv_split.Output("q_out")[0]
        k = qkv_split.Output("k_out")[0]
        v = qkv_split.Output("v_out")[0]
        write_cache_kv = ir.PassDesc.OP.write_cache_kv(cache_kv=cache_kv, input_k=k, input_v=v, sequence_lengths=seq_len_encoder)
        attention = ir.PassDesc.OP.variable_length_memory_efficient_attention(key=k, kv_seq_lens=seq_len_encoder, mask=mask, query=q, seq_lens=seq_len_encoder, value=v)
        
        transpose_remove_padding = ir.PassDesc.OP.transpose_remove_padding(input=attention, padding_offset=padding_offset, seq_lens=seq_len_encoder)
        matmul_0 = ir.PassDesc.OP.matmul_v2(X=transpose_remove_padding, Y=out_proj_weight)
        
        allreduce = ir.PassDesc.OP.c_allreduce_sum(X=matmul_0)
        
        rms_norm_1 = ir.PassDesc.OP.rms_norm(norm_weight=ffn_in_scale, residual=rms_norm_0.Output("residual_out")[0], x=allreduce)
        matmul_1 = ir.PassDesc.OP.matmul_v2(X=rms_norm_1.Output("out"), Y=ffn1_weight)
        fused_bias_act = ir.PassDesc.OP.fused_bias_act(x=matmul_1)
        
        matmul_2 = ir.PassDesc.OP.matmul_v2(X=fused_bias_act, Y=ffn2_weight)
        hidden = ir.PassDesc.OP.c_allreduce_sum(X=matmul_2)
        residual_out = rms_norm_1.Output("residual_out")[0]
        
        encode_rotary_qk = ir.PassDesc.OP.encode_rotary_qk(kv=k, q=q, rotary_emb=rotary_emb, seq_lens=seq_len_encoder)
        rotary_kv_out = encode_rotary_qk.Output("rotary_kv_out")[0]
        rotary_q_out = encode_rotary_qk.Output("rotary_q_out")[0]
        
        return write_cache_kv, q, k, v, hidden, residual_out, rotary_kv_out, rotary_q_out
        
    def replace(x, residual, input_ids, padding_offset, seq_len_encoder, cache_kv, mask, rotary_emb, ln_scale, qkv_weight, out_proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight):
        llama_layer = llama_paralle_layer_adaptor(x, residual, input_ids, padding_offset, seq_len_encoder, cache_kv, mask, rotary_emb, ln_scale, qkv_weight, out_proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight)
        
        return (llama_layer[3],
            llama_layer[4],
            llama_layer[5],
            llama_layer[6],
            llama_layer[0],
            llama_layer[7],
            llama_layer[1],
            llama_layer[2])

    return pattern, replace
    

@ir.RegisterPass
def llama_fuse_attention_dynamic_first_parallel_layer():
    def pattern(x, input_ids, padding_offset, seq_len_encoder, cache_kv, mask, rotary_emb, ln_scale, qkv_weight, out_proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight):
        rms_norm_0 = ir.PassDesc.OP.rms_norm(norm_weight=ln_scale, x=x)
        qkv = ir.PassDesc.OP.matmul_v2(X=rms_norm_0.Output("out"), Y=qkv_weight)
        qkv_split = ir.PassDesc.OP.qkv_transpose_split(input_ids=input_ids, padding_offset=padding_offset, qkv=qkv, seq_lens=seq_len_encoder)
        q = qkv_split.Output("q_out")[0]
        k = qkv_split.Output("k_out")[0]
        v = qkv_split.Output("v_out")[0]
        write_cache_kv = ir.PassDesc.OP.write_cache_kv(cache_kv=cache_kv, input_k=k, input_v=v, sequence_lengths=seq_len_encoder)
        attention = ir.PassDesc.OP.variable_length_memory_efficient_attention(key=k, kv_seq_lens=seq_len_encoder, mask=mask, query=q, seq_lens=seq_len_encoder, value=v)
        
        transpose_remove_padding = ir.PassDesc.OP.transpose_remove_padding(input=attention, padding_offset=padding_offset, seq_lens=seq_len_encoder)
        matmul_0 = ir.PassDesc.OP.matmul_v2(X=transpose_remove_padding, Y=out_proj_weight)
        
        allreduce = ir.PassDesc.OP.c_allreduce_sum(X=matmul_0)
        
        rms_norm_1 = ir.PassDesc.OP.rms_norm(norm_weight=ffn_in_scale, residual=x, x=allreduce)
        matmul_1 = ir.PassDesc.OP.matmul_v2(X=rms_norm_1.Output("out"), Y=ffn1_weight)
        fused_bias_act = ir.PassDesc.OP.fused_bias_act(x=matmul_1)
        
        matmul_2 = ir.PassDesc.OP.matmul_v2(X=fused_bias_act, Y=ffn2_weight)
        hidden = ir.PassDesc.OP.c_allreduce_sum(X=matmul_2)
        residual_out = rms_norm_1.Output("residual_out")[0]
        
        encode_rotary_qk = ir.PassDesc.OP.encode_rotary_qk(kv=k, q=q, rotary_emb=rotary_emb, seq_lens=seq_len_encoder)
        rotary_kv_out = encode_rotary_qk.Output("rotary_kv_out")[0]
        rotary_q_out = encode_rotary_qk.Output("rotary_q_out")[0]
        
        return write_cache_kv, q, k, v, hidden, residual_out, rotary_kv_out, rotary_q_out
        
    def replace(x, input_ids, padding_offset, seq_len_encoder, cache_kv, mask, rotary_emb, ln_scale, qkv_weight, out_proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight):
        llama_layer = llama_paralle_layer_adaptor(x, None, input_ids, padding_offset, seq_len_encoder, cache_kv, mask, rotary_emb, ln_scale, qkv_weight, out_proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight)
        
        return (llama_layer[3],
            llama_layer[4],
            llama_layer[5],
            llama_layer[6],
            llama_layer[0],
            llama_layer[7],
            llama_layer[1],
            llama_layer[2])

    return pattern, replace

@ir.RegisterPass
def llama65B_fuse_attention_dynamic_parallel_layer():
    def pattern(x, residual, input_ids, padding_offset, seq_len_encoder, cache_kv, mask, rotary_emb, ln_scale, qkv_weight, out_proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight):
        rms_norm_0 = ir.PassDesc.OP.rms_norm(norm_weight=ln_scale, residual=residual, x=x)
        qkv = ir.PassDesc.OP.matmul_v2(X=rms_norm_0.Output("out"), Y=qkv_weight)
        qkv_split = ir.PassDesc.OP.qkv_transpose_split(input_ids=input_ids, padding_offset=padding_offset, qkv=qkv, seq_lens=seq_len_encoder)
        q = qkv_split.Output("q_out")[0]
        k = qkv_split.Output("k_out")[0]
        v = qkv_split.Output("v_out")[0]
        scale1 = ir.PassDesc.OP.scale(X=seq_len_encoder)
        write_cache_kv = ir.PassDesc.OP.write_cache_kv(cache_kv=cache_kv, input_k=k, input_v=v, sequence_lengths=scale1)
        scale2 = ir.PassDesc.OP.scale(X=seq_len_encoder)
        attention = ir.PassDesc.OP.variable_length_memory_efficient_attention(key=k, kv_seq_lens=scale2, mask=mask, query=q, seq_lens=seq_len_encoder, value=v)
        
        transpose_remove_padding = ir.PassDesc.OP.transpose_remove_padding(input=attention, padding_offset=padding_offset, seq_lens=seq_len_encoder)
        matmul_0 = ir.PassDesc.OP.matmul_v2(X=transpose_remove_padding, Y=out_proj_weight)
        
        allreduce = ir.PassDesc.OP.c_allreduce_sum(X=matmul_0)
        
        rms_norm_1 = ir.PassDesc.OP.rms_norm(norm_weight=ffn_in_scale, residual=rms_norm_0.Output("residual_out")[0], x=allreduce)
        matmul_1 = ir.PassDesc.OP.matmul_v2(X=rms_norm_1.Output("out"), Y=ffn1_weight)
        fused_bias_act = ir.PassDesc.OP.fused_bias_act(x=matmul_1)
        
        matmul_2 = ir.PassDesc.OP.matmul_v2(X=fused_bias_act, Y=ffn2_weight)
        hidden = ir.PassDesc.OP.c_allreduce_sum(X=matmul_2)
        residual_out = rms_norm_1.Output("residual_out")[0]
        
        encode_rotary_qk = ir.PassDesc.OP.encode_rotary_qk(kv=k, q=q, rotary_emb=rotary_emb, seq_lens=seq_len_encoder)
        rotary_kv_out = encode_rotary_qk.Output("rotary_kv_out")[0]
        rotary_q_out = encode_rotary_qk.Output("rotary_q_out")[0]
        
        return write_cache_kv, q, k, v, hidden, residual_out, rotary_kv_out, rotary_q_out
        
    def replace(x, residual, input_ids, padding_offset, seq_len_encoder, cache_kv, mask, rotary_emb, ln_scale, qkv_weight, out_proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight):
        llama_layer = llama_paralle_layer_adaptor(x, residual, input_ids, padding_offset, seq_len_encoder, cache_kv, mask, rotary_emb, ln_scale, qkv_weight, out_proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight)
        
        return (llama_layer[3],
            llama_layer[4],
            llama_layer[5],
            llama_layer[6],
            llama_layer[0],
            llama_layer[7],
            llama_layer[1],
            llama_layer[2])

    return pattern, replace
    

@ir.RegisterPass
def llama65B_fuse_attention_dynamic_first_parallel_layer():
    def pattern(x, input_ids, padding_offset, seq_len_encoder, cache_kv, mask, rotary_emb, ln_scale, qkv_weight, out_proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight):
        rms_norm_0 = ir.PassDesc.OP.rms_norm(norm_weight=ln_scale, x=x)
        qkv = ir.PassDesc.OP.matmul_v2(X=rms_norm_0.Output("out"), Y=qkv_weight)
        qkv_split = ir.PassDesc.OP.qkv_transpose_split(input_ids=input_ids, padding_offset=padding_offset, qkv=qkv, seq_lens=seq_len_encoder)
        q = qkv_split.Output("q_out")[0]
        k = qkv_split.Output("k_out")[0]
        v = qkv_split.Output("v_out")[0]
        scale1 = ir.PassDesc.OP.scale(X=seq_len_encoder)
        write_cache_kv = ir.PassDesc.OP.write_cache_kv(cache_kv=cache_kv, input_k=k, input_v=v, sequence_lengths=scale1)
        scale2 = ir.PassDesc.OP.scale(X=seq_len_encoder)
        attention = ir.PassDesc.OP.variable_length_memory_efficient_attention(key=k, kv_seq_lens=scale2, mask=mask, query=q, seq_lens=seq_len_encoder, value=v)
        
        transpose_remove_padding = ir.PassDesc.OP.transpose_remove_padding(input=attention, padding_offset=padding_offset, seq_lens=seq_len_encoder)
        matmul_0 = ir.PassDesc.OP.matmul_v2(X=transpose_remove_padding, Y=out_proj_weight)
        
        allreduce = ir.PassDesc.OP.c_allreduce_sum(X=matmul_0)
        
        rms_norm_1 = ir.PassDesc.OP.rms_norm(norm_weight=ffn_in_scale, residual=x, x=allreduce)
        matmul_1 = ir.PassDesc.OP.matmul_v2(X=rms_norm_1.Output("out"), Y=ffn1_weight)
        fused_bias_act = ir.PassDesc.OP.fused_bias_act(x=matmul_1)
        
        matmul_2 = ir.PassDesc.OP.matmul_v2(X=fused_bias_act, Y=ffn2_weight)
        hidden = ir.PassDesc.OP.c_allreduce_sum(X=matmul_2)
        residual_out = rms_norm_1.Output("residual_out")[0]
        
        encode_rotary_qk = ir.PassDesc.OP.encode_rotary_qk(kv=k, q=q, rotary_emb=rotary_emb, seq_lens=seq_len_encoder)
        rotary_kv_out = encode_rotary_qk.Output("rotary_kv_out")[0]
        rotary_q_out = encode_rotary_qk.Output("rotary_q_out")[0]
        
        return write_cache_kv, q, k, v, hidden, residual_out, rotary_kv_out, rotary_q_out
        
    def replace(x, input_ids, padding_offset, seq_len_encoder, cache_kv, mask, rotary_emb, ln_scale, qkv_weight, out_proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight):
        llama_layer = llama_paralle_layer_adaptor(x, None, input_ids, padding_offset, seq_len_encoder, cache_kv, mask, rotary_emb, ln_scale, qkv_weight, out_proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight)
        
        return (llama_layer[3],
            llama_layer[4],
            llama_layer[5],
            llama_layer[6],
            llama_layer[0],
            llama_layer[7],
            llama_layer[1],
            llama_layer[2])

    return pattern, replace


@ir.RegisterPass
def llama_lmhead():
    def pattern(x, norm_weight, matmul_weight):
        rms_norm_ = ir.PassDesc.OP.rms_norm(x=x, norm_weight=norm_weight)
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

@ir.RegisterPass
def llama_fuse_attention_dynamic_parallel_layer_be61d():
    def pattern(x, residual, input_ids, padding_offset, seq_len_encoder, sequence_lengths, kv_seq_lens, cache_kv, mask, rotary_emb, ln_scale, qkv_weight, out_proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight):
        rms_norm_0 = ir.PassDesc.OP.rms_norm(norm_weight=ln_scale, residual=residual, x=x)
        qkv = ir.PassDesc.OP.matmul_v2(X=rms_norm_0.Output("out"), Y=qkv_weight)
        qkv_split = ir.PassDesc.OP.qkv_transpose_split(input_ids=input_ids, padding_offset=padding_offset, qkv=qkv, seq_lens=seq_len_encoder)
        q = qkv_split.Output("q_out")[0]
        k = qkv_split.Output("k_out")[0]
        v = qkv_split.Output("v_out")[0]
        write_cache_kv = ir.PassDesc.OP.write_cache_kv(cache_kv=cache_kv, input_k=k, input_v=v, sequence_lengths=sequence_lengths)
        attention = ir.PassDesc.OP.variable_length_memory_efficient_attention(key=k, kv_seq_lens=kv_seq_lens, mask=mask, query=q, seq_lens=seq_len_encoder, value=v)
        
        transpose_remove_padding = ir.PassDesc.OP.transpose_remove_padding(input=attention, padding_offset=padding_offset, seq_lens=seq_len_encoder)
        matmul_0 = ir.PassDesc.OP.matmul_v2(X=transpose_remove_padding, Y=out_proj_weight)
        
        allreduce = ir.PassDesc.OP.c_allreduce_sum(X=matmul_0)
        
        rms_norm_1 = ir.PassDesc.OP.rms_norm(norm_weight=ffn_in_scale, residual=rms_norm_0.Output("residual_out")[0], x=allreduce)
        matmul_1 = ir.PassDesc.OP.matmul_v2(X=rms_norm_1.Output("out"), Y=ffn1_weight)
        fused_bias_act = ir.PassDesc.OP.fused_bias_act(x=matmul_1)
        
        matmul_2 = ir.PassDesc.OP.matmul_v2(X=fused_bias_act, Y=ffn2_weight)
        hidden = ir.PassDesc.OP.c_allreduce_sum(X=matmul_2)
        residual_out = rms_norm_1.Output("residual_out")[0]
        
        encode_rotary_qk = ir.PassDesc.OP.encode_rotary_qk(kv=k, q=q, rotary_emb=rotary_emb, seq_lens=seq_len_encoder)
        rotary_kv_out = encode_rotary_qk.Output("rotary_kv_out")[0]
        rotary_q_out = encode_rotary_qk.Output("rotary_q_out")[0]
        
        return write_cache_kv, q, k, v, hidden, residual_out, rotary_kv_out, rotary_q_out
        
    def replace(x, residual, input_ids, padding_offset, seq_len_encoder, sequence_lengths, kv_seq_lens, cache_kv, mask, rotary_emb, ln_scale, qkv_weight, out_proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight):
        llama_layer = llama_paralle_layer_adaptor(x, residual, input_ids, padding_offset, seq_len_encoder, cache_kv, mask, rotary_emb, ln_scale, qkv_weight, out_proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight)
        
        return (llama_layer[3],
            llama_layer[4],
            llama_layer[5],
            llama_layer[6],
            llama_layer[0],
            llama_layer[7],
            llama_layer[1],
            llama_layer[2])

    return pattern, replace
    

@ir.RegisterPass
def llama_fuse_attention_dynamic_first_parallel_layer_be61d():
    # def pattern(x, input_ids, padding_offset, seq_len_encoder, cache_kv, mask, rotary_emb, ln_scale, qkv_weight, out_proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight):
    def pattern(x, input_ids, padding_offset, seq_len_encoder, sequence_lengths, kv_seq_lens, mask, cache_kv, rotary_emb, ln_scale, qkv_weight, out_proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight):
        rms_norm_0 = ir.PassDesc.OP.rms_norm(norm_weight=ln_scale, x=x)
        qkv = ir.PassDesc.OP.matmul_v2(X=rms_norm_0.Output("out"), Y=qkv_weight)
        qkv_split = ir.PassDesc.OP.qkv_transpose_split(input_ids=input_ids, padding_offset=padding_offset, qkv=qkv, seq_lens=seq_len_encoder)
        q = qkv_split.Output("q_out")[0]
        k = qkv_split.Output("k_out")[0]
        v = qkv_split.Output("v_out")[0]
        write_cache_kv = ir.PassDesc.OP.write_cache_kv(cache_kv=cache_kv, input_k=k, input_v=v, sequence_lengths=sequence_lengths)

        attention = ir.PassDesc.OP.variable_length_memory_efficient_attention(key=k, kv_seq_lens=kv_seq_lens, mask=mask, query=q, seq_lens=seq_len_encoder, value=v)
        
        transpose_remove_padding = ir.PassDesc.OP.transpose_remove_padding(input=attention, padding_offset=padding_offset, seq_lens=seq_len_encoder)
        matmul_0 = ir.PassDesc.OP.matmul_v2(X=transpose_remove_padding, Y=out_proj_weight)
        
        allreduce = ir.PassDesc.OP.c_allreduce_sum(X=matmul_0)
        
        rms_norm_1 = ir.PassDesc.OP.rms_norm(norm_weight=ffn_in_scale, residual=x, x=allreduce)
        matmul_1 = ir.PassDesc.OP.matmul_v2(X=rms_norm_1.Output("out"), Y=ffn1_weight)
        fused_bias_act = ir.PassDesc.OP.fused_bias_act(x=matmul_1)
        
        matmul_2 = ir.PassDesc.OP.matmul_v2(X=fused_bias_act, Y=ffn2_weight)
        hidden = ir.PassDesc.OP.c_allreduce_sum(X=matmul_2)
        residual_out = rms_norm_1.Output("residual_out")[0]
        
        encode_rotary_qk = ir.PassDesc.OP.encode_rotary_qk(kv=k, q=q, rotary_emb=rotary_emb, seq_lens=seq_len_encoder)
        rotary_kv_out = encode_rotary_qk.Output("rotary_kv_out")[0]
        rotary_q_out = encode_rotary_qk.Output("rotary_q_out")[0]
        
        return write_cache_kv, q, k, v, hidden, residual_out, rotary_kv_out, rotary_q_out
        
    def replace(x, input_ids, padding_offset, seq_len_encoder, sequence_lengths, kv_seq_lens, mask, cache_kv, rotary_emb, ln_scale, qkv_weight, out_proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight):
        llama_layer = llama_paralle_layer_adaptor(x, None, input_ids, padding_offset, seq_len_encoder, cache_kv, mask, rotary_emb, ln_scale, qkv_weight, out_proj_weight, ffn_in_scale, ffn1_weight, ffn2_weight)
        
        return (llama_layer[3],
            llama_layer[4],
            llama_layer[5],
            llama_layer[6],
            llama_layer[0],
            llama_layer[7],
            llama_layer[1],
            llama_layer[2])

    return pattern, replace

@ir.RegisterPass
def replace_embeding():
    def pattern(x, y):
        return ir.PassDesc.OP.lookup_table_v2(Ids=x, W=y)

    def replace(x, y):
        return ir.PassDesc.OP.atb_embeding(X=x, Y=y)
    