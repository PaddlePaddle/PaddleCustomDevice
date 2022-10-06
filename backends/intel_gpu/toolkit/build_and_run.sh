#!/bin/bash
export PLUGIN_VERBOSE=15
org=`pwd`
#LAGS_allocator_strategy=auto_growth
#FLAGS_allocator_strategy=naive_best_fit
#export DNNL_VERBOSE=1
#clang-format -i -style=file runtime/runtime.cc


cd build
make -j4 


if [ ! $? -eq 0 ]; then
exit;
fi

pip install  --force-reinstall  dist/paddle_custom_intel_gpu-0.0.1-cp39-cp39-linux_x86_64.whl


if [ ! $? -eq 0 ]; then

exit;
fi

#python ../tests/simple_test.py



cd $org

#export PYTHONPATH=/home/pawel/work/PaddleIntelGPUDevice/python/
export PYTHONPATH=`pwd`/../../python/
export PYTHONPATH=$PYTHONPATH:$PYTHONPATH/tests/:$PYTHONPATH/tests/unittests


#python tests/unittests/test_elementwise_mul_op.py
python tests/unittests/test_elementwise_mul_op_fp32.py
#python tests/unittests/test_transpose_op_gpu.py
#python tests/unittests/test_transpose_op.py
#python tests/assign_op.py
#python tests/unittests/test_assign_value_op.py

#python tests/unittests/test_fill_constant_op.py

