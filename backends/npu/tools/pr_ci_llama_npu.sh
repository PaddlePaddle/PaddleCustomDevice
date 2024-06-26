# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Baseline
train_loss=10.627841186523437
train_samples_per_second=2.4003


function check_loss() {
  pr_train_loss=`grep "train_loss" /paddle/PaddleNLP/llm/llama/log_llama_ci/workerlog.0|tail -1|awk '{print $NF}'`
  if [ "$train_loss" = "$pr_train_loss" ]; then
      echo "train_loss is Same"
  else
      echo "train_loss is different"
      exit 8
  fi
}


function check_train() {
  pr_train_samples_per_second=`grep "train_samples_per_second" workerlog.0 |tail -1|cat -v|awk '{print $NF}'|awk -F '^' '{print $1}'`
  int_train=`echo |awk "{print ${train_samples_per_second} * 100}"`
  pr_train=`echo |awk "{print ${pr_train_samples_per_second} * 100}"`
  diff_train=`echo |awk "{print int(${int_train} - ${pr_train})}"`
  if [ $diff_train -le 2 ]; then
      echo "train_samples_per_second is less 2%"
  else
      echo "train_samples_per_second is greater than 2%"
      exit 8
  fi
}


function build() {
  bash backends/npu/tools/compile.sh
  if [[ "$?" != "0" ]];then
      exit 7;
  fi
}


function install_depend() {
  pip install /paddle/backends/npu/build/dist/*.whl
  pip install -r /paddle/PaddleNLP/requirements.txt
  cd /paddle/PaddleNLP/llm/llama
  mkdir pre-data
  cd pre-data
  wget -q https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.bin
  wget -q https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.idx
}


function open_lock_seed() {
  echo "lock_seed_flag : $lock_seed_flag"
  if [[ ${lock_seed_flag} =~ "open_lock_seed" ]];then
      export npu_deterministic=true
      export ACL_OP_DETERMINISTIC=true
      export ACL_OPT_DETERMINISTIC=true
      export HCCL_DETERMINISTIC=true
      echo "npu_deterministic : $npu_deterministic   ACL_OP_DETERMINISTIC : $ACL_OP_DETERMINISTIC   ACL_OPT_DETERMINISTIC : $ACL_OPT_DETERMINISTIC   HCCL_DETERMINISTIC : $HCCL_DETERMINISTIC"
  fi
}


function run_test() {

  lock_seed_flag=${1:-close}
  set -x
  ps aux | grep run_pretrain.py | grep -v grep | awk '{print $2}' | xargs kill -9
  rm -rf ./log_llama_ci
  rm -rf output
  export PYTHONPATH=../../:$PYTHONPATH
  export MC2=1
  export GLOG_v=0
  export FLAGS_npu_storage_format=1
  export HCCL_INTRA_PCIE_EHABLE=0
  export HCCL_INTRA_ROCE_ENABLE=1
  export FLAGS_allocator_strategy=naive_best_fit
  export FLAGS_NPU_MC2=1
  export MC2_Recompute=1
  unset PADDLE_TRAINER_ENDPOINTS
  unset DISTRIBUTED_TRAINER_ENDPOINTS
  export FLAGS_use_stride_kernel=0
  export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
  export MULTI_STREAM_MEMORY_REUSE=1
  source /usr/local/Ascend/ascend-toolkit/set_env.sh

  cd /paddle/PaddleNLP/llm/llama
  python -u  -m paddle.distributed.launch \
    --log_dir "./log_llama_ci" \
    --devices 0,1,2,3 \
    ../run_pretrain.py \
    --model_name_or_path "meta-llama/Llama-2-7b" \
    --tokenizer_name_or_path "meta-llama/Llama-2-7b" \
    --input_dir "./pre-data" \
    --output_dir "./output" \
    --split 949,50,1 \
    --max_seq_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size 1 \
    --use_flash_attention 1 \
    --use_fused_rms_norm 1 \
    --virtual_pp_degree 1 \
    --learning_rate 0.00001 \
    --min_learning_rate 0.000001 \
    --max_steps 10 \
    --decay_steps 2000 \
    --save_steps 2000 \
    --seed 100 \
    --weight_decay 0.01 \
    --warmup_steps 20 \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --dataloader_num_workers 1 \
    --eval_steps 1001 \
    --tensor_parallel_degree 2 \
    --disable_tqdm true \
    --continue_training 0 \
    --do_train \
    --device "npu" \
    --enable_linear_fused_grad_add false \
    --fuse_attention_qkv true \
    --fuse_attention_ffn true \
    --use_fused_rope true \
    --recompute_use_reentrant true \
    --data_cache "./data_cache" \
    --bf16 \
    --fp16_opt_level "O2" \
    --amp_master_grad \
    --load_sharded_model true \
    --save_sharded_model true \
    --pipeline_parallel_degree 1 \
    --ignore_data_skip 0 \
    --force_reshard_pp true \
    --tensor_parallel_config "enable_mp_async_allreduce enable_mp_skip_c_identity" \
    --sequence_parallel 1 \
    --pipeline_parallel_config "disable_partial_send_recv" \
    --sharding "stage1" \
    --sharding_parallel_degree 2
}

function main() {
  build
  install_depend
  run_test
  open_lock_seed
  run_test 
}
main
