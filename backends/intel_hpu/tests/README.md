# How to run all unit tests for PR testing
# setup the PYTHONPATH env
export PYTHONPATH=/workspace/pdpd_ci/PaddleCustomDevice/python/:/workspace/pdpd_ci/PaddleCustomDevice/Paddle/test/legacy_test/

# common cmdline
python pr-test-run.py --test_path /workspace/pdpd_ci/PaddleCustomDevice/backends/intel_hpu/tests/unittests/

# to run all test cases under the --test_path folder with xml output
python pr-test-run.py --test_path /workspace/pdpd_ci/PaddleCustomDevice/backends/intel_hpu/tests/unittests/ --junit test_result.xml --platform gaudi2

# to run special test cases
python pr-test-run.py --test_path /workspace/pdpd_ci/PaddleCustomDevice/backends/intel_hpu/tests/unittests/ --junit test_result.xml --platform gaudi2 --k test_abs_op
python pr-test-run.py --test_path /workspace/pdpd_ci/PaddleCustomDevice/backends/intel_hpu/tests/unittests/ --junit test_result.xml --platform gaudi2 --k test_abs_op.py

#how to run all, stable, or unstable test cases
stable: all passed test cases suits
unstable: test sub suites which have failed cases
python pr-test-run.py --test_path /workspace_pdpd/repo/PaddleCustomDevice/backends/intel_hpu/tests/unittests/ --junit test_result.xml --platform gaudi2
python pr-test-run.py --test_path /workspace_pdpd/repo/PaddleCustomDevice/backends/intel_hpu/tests/unittests/ --filter all --junit test_result.xml --platform gaudi2
python pr-test-run.py --test_path /workspace_pdpd/repo/PaddleCustomDevice/backends/intel_hpu/tests/unittests/ --filter stable --junit test_result.xml --platform gaudi2
python pr-test-run.py --test_path /workspace_pdpd/repo/PaddleCustomDevice/backends/intel_hpu/tests/unittests/ --filter unstable --junit test_result.xml --platform gaudi2

#force exit if run into any failed case
python pr-test-run.py --test_path /workspace_pdpd/repo/PaddleCustomDevice/backends/intel_hpu/tests/unittests/ --filter stable --junit test_result.xml --platform gaudi2 --force_exit
