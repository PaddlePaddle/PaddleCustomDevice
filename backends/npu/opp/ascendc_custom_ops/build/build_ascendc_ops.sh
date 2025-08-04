#!/bin/bash
# -*- coding: utf-8 -*-

src=${current_script_dir}/../src/ops/ascendc
dst=${current_script_dir}/custom_project

function create_empty_custom_project(){
    cd ${current_script_dir}
    rm -rf ${dst}
    ${msopgen} gen -i ir_demo.json -f onnx \
        -c ai_core-ascend310p,ai_core-ascend910,ai_core-ascend910b -lan cpp -out ${dst}
    rm ${dst}/framework/onnx_plugin/*.cc
    rm ${dst}/op_host/*.h
    rm ${dst}/op_host/*.cpp
    rm ${dst}/op_kernel/*.cpp
}

function release_framework_onnx(){
    cd ${src}/framework/onnx_plugin
    # 如需控制哪些文件发布，可以按照字母序列举具体文件
    local files=(
        set_mask_value_plugin.cpp
        set_stop_value_multi_ends_plugin.cpp
        set_value_by_flags_and_idx_plugin.cpp
        token_penalty_multi_scores_plugin.cpp
    )
    cp ${files[@]} ${dst}/framework/onnx_plugin
}

function release_op_host(){
    cd ${src}/op_host
    local files=(
        set_value_by_flags_and_idx_tiling.h
        set_value_by_flags_and_idx.cpp

        set_value_by_flags_and_idx_v2_tiling.h
        set_value_by_flags_and_idx_v2.cpp

        set_stop_value_multi_ends_tiling.h
        set_stop_value_multi_ends.cpp

        set_stop_value_multi_ends_v2_tiling.h
        set_stop_value_multi_ends_v2.cpp

        set_stop_value_multi_seqs_tiling.h
        set_stop_value_multi_seqs.cpp

        set_mask_value_tiling.h
        set_mask_value.cpp

        token_penalty_multi_scores_tiling.h
        token_penalty_multi_scores.cpp

        token_penalty_multi_scores_v2_tiling.h
        token_penalty_multi_scores_v2.cpp

        token_penalty_multi_scores_with_stop_seqs_tiling.h
        token_penalty_multi_scores_with_stop_seqs.cpp

        update_inputs_tiling.h
        update_inputs.cpp

        get_max_len_tiling.h
        get_max_len.cpp

        rebuild_padding_tiling.h
        rebuild_padding.cpp

        get_padding_offset_tiling.h
        get_padding_offset.cpp

        step_paddle_tiling.h
        step_paddle.cpp
    )
    cp ${files[@]} ${dst}/op_host
}

function release_op_kernel(){
    cd ${src}/op_kernel
    local files=(
        set_value_by_flags_and_idx.cpp
        set_value_by_flags_and_idx_v2.cpp
        set_stop_value_multi_ends.cpp
        set_stop_value_multi_ends_v2.cpp
        set_stop_value_multi_seqs.cpp
        set_mask_value.cpp
        token_penalty_multi_scores.cpp
        token_penalty_multi_scores_v2.cpp
        token_penalty_multi_scores_with_stop_seqs.cpp
        update_inputs.cpp
        get_max_len.cpp
        rebuild_padding.cpp
        get_padding_offset.cpp
        step_paddle.cpp
    )
    cp ${files[@]} ${dst}/op_kernel
}

function revise_settings(){
    cd ${dst}
    sed -i "s#/usr/local/Ascend/latest#${local_toolkit}#g" CMakePresets.json
    sed -i "s#\"value\": \"customize\"#\"value\": \"aie_ascendc\"#g" CMakePresets.json
    sed -i "s#\"value\": \"True\"#\"value\": \"False\"#g" CMakePresets.json

    local line_num=$(grep -Fn "ENABLE_SOURCE_PACKAGE" CMakePresets.json | cut -d : -f 1)
    local offset_line_num=$((line_num+2))
    sed -i "${offset_line_num}s#\"value\": \"False\"#\"value\": \"True\"#g" CMakePresets.json
}

function build_and_install(){
    cd ${dst}
    bash build.sh
    bash ${dst}/build_out/*.run --install-path=${current_script_dir}
}

function build_ascendc_ops(){
    ori_path=${PWD}
    
    create_empty_custom_project
    release_framework_onnx
    release_op_host
    release_op_kernel
    revise_settings
    build_and_install
    cd ${ori_path}
}

build_ascendc_ops
