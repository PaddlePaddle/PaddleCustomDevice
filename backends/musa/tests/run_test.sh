
SCRIPT_DIR=$(dirname "$(realpath "$0")")
LEGACY_TEST_PATH="${SCRIPT_DIR}/../../../Paddle/test/legacy_test"
export PYTHONPATH="${LEGACY_TEST_PATH}:${PYTHONPATH}"


mkdir -p build || { echo "ERROR: Failed to create build directory"; exit 1; }
cd build || { echo "ERROR: Failed to enter build directory"; exit 1; }

cmake .. || { echo "ERROR: CMake configuration failed"; exit 1; }

make -j$(nproc) || { echo "ERROR: Build failed"; exit 1; }

ctest --output-on-failure -V || { 
    echo "ERROR: Tests failed!" >&2
    echo "Exit code: $?" >&2
    exit 1
}

echo "=== All tests passed successfully ==="
cd - > /dev/null

