# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

set -ex

WORKSPACE=`pwd`
echo "Install whl"
pip install safetensors==0.6.2
python -m pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
python -c "import paddle; print(paddle.__version__)"
python -c "import paddle; print(paddle.version.commit)"
python -m pip install lxml numpy==1.26.4

echo "custom_ops install"
cd ${WORKSPACE}/PaddleCustomDevice/backends/intel_hpu/custom_ops
python setup.py install

echo "Start build"
cd ${WORKSPACE}/PaddleCustomDevice/backends/intel_hpu
mkdir -p build && cd build
cmake .. 
make -j $(nproc)

# Update lib.so
rm -rf  /usr/lib/habanalabs/libcustom_tpc_perf_lib.so 
wget --no-proxy -q https://paddle-ci.cdn.bcebos.com/libcustom_tpc_perf_lib.so -P /usr/lib/habanalabs/ 
cp /usr/lib/habanalabs/libcustom_tpc_perf_lib.so ${WORKSPACE}/PaddleCustomDevice/backends/intel_hpu/build/

python -m pip install --force-reinstall -U dist/paddle*.whl
export PYTHONPATH=/workspace/PaddleCustomDevice/python:/workspace/PaddleCustomDevice/python/tests:$PYTHONPATH

echo "Start Test"
python ${WORKSPACE}/PaddleCustomDevice/backends/intel_hpu/tests/pr-test-run.py --test_path ${WORKSPACE}/PaddleCustomDevice/backends/intel_hpu/tests/unittests/ --junit ${WORKSPACE}/ci.xml --filter stable --platform gaudi2 --force_exit
python ${WORKSPACE}/PaddleCustomDevice/backends/intel_hpu/tools/testresult_analyse.py --input_file ${WORKSPACE}/ci.xml >>${WORKSPACE}/ci.log
cat ${WORKSPACE}/ci.log
