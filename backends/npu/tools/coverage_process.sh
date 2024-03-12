# install lcov
wget -P /home https://paddle-ci.cdn.bcebos.com/coverage/lcov-1.16.tar.gz --no-proxy --no-check-certificate || exit 101
tar -xf /home/lcov-1.16.tar.gz -C /
cd /lcov-1.16
make install

lcov --ignore-errors gcov --capture -d ./ -o coverage.info --rc lcov_branch_coverage=0

function gen_full_html_report() {
    lcov --extract coverage.info \
        '/paddle/backends/npu/custom_op/*' \
        '/paddle/backends/npu/kernels/*' \
        '/paddle/backends/npu/runtime/*' \
        -o coverage-full.tmp \
        --rc lcov_branch_coverage=0

    mv -f coverage-full.tmp coverage-full.info
}

gen_full_html_report

genhtml coverage-full.info --output-directory cpp-coverage-full