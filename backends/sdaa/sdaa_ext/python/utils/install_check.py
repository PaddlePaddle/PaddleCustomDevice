# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.

import paddle
import paddle.base as base
import logging
from ..version import version
from ..version import version_query

pretty_print = True
try:
    import prettytable
except ImportError:
    pretty_print = False

CUSTOM_DEVICE = "sdaa"


def _version_check():
    if pretty_print:
        info = ["Dependence", "Compilation Version", "Current Version"]
        version_table = prettytable.PrettyTable(info)
    # get all dependences name
    except_list = ["paddle_commit", "paddle_sdaa_commit"]
    check_list = [
        var
        for var in dir(version)
        if not var.startswith("__") and var not in except_list
    ]
    joined_name = ""
    for name in check_list:
        compile_version = getattr(version, name)
        runtime_version_func = getattr(version_query, name + "_version")
        runtime_version = runtime_version_func()
        if compile_version != runtime_version:
            joined_name += name + "/"
            if pretty_print:
                version_table.add_row([name, compile_version, runtime_version])
            else:
                logging.warning(
                    "{} Compilation Version: {}, Runtime Version: {}".format(
                        name, compile_version, runtime_version
                    )
                )
    if 0 != len(joined_name):
        logging.warning(
            "Current development environment has different versions of {} "
            "compared to the version it was compiled with.\n".format(joined_name)
        )
        if pretty_print:
            print(version_table)


def _is_sdaa_available():
    try:
        sdaa_device_ids = range(base.core.get_custom_device_count(CUSTOM_DEVICE))
        assert len(sdaa_device_ids) > 0
        sdaa_places = [
            paddle.CustomPlace(CUSTOM_DEVICE, did) for did in sdaa_device_ids
        ]
        return (True, sdaa_places)

    except Exception as e:
        logging.warning(
            "You are using SDAA PaddlePaddle, but there is no SDAA Device "
            "detected on your machine.\n"
        )
        return (False, [])


def _run_static_single(place):
    paddle.enable_static()
    with paddle.static.scope_guard(paddle.static.Scope()):
        train_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        startup_prog.random_seed = 1
        with paddle.static.program_guard(train_prog, startup_prog):
            input, out, weight = paddle.utils.install_check._simple_network()
            param_grads = paddle.static.append_backward(
                out, parameter_list=[weight.name]
            )[0]
        exe = paddle.static.Executor(place)
        exe.run(startup_prog)
        exe.run(
            train_prog,
            feed={input.name: paddle.utils.install_check._prepare_data()},
            fetch_list=[out.name, param_grads[1].name],
        )
    paddle.disable_static()


def _run_dygraph_single(place):
    paddle.disable_static()
    base.framework._set_expected_place(place)
    weight_attr = paddle.ParamAttr(
        name="weight", initializer=paddle.nn.initializer.Constant(value=0.5)
    )
    bias_attr = paddle.ParamAttr(
        name="bias", initializer=paddle.nn.initializer.Constant(value=1.0)
    )
    linear = paddle.nn.Linear(2, 4, weight_attr=weight_attr, bias_attr=bias_attr)
    input_np = paddle.utils.install_check._prepare_data()
    input_tensor = paddle.to_tensor(input_np)
    linear_out = linear(input_tensor)
    out = paddle.tensor.sum(linear_out)
    out = paddle.nn.functional.sigmoid(linear_out)
    out.backward()
    opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=linear.parameters())
    opt.step()


def _test_on_one_device(place):
    # set fallback flag to False in order to throw exception when sdaa_kernel fallback to CPU
    enable_fallback_str = "FLAGS_enable_api_kernel_fallback"
    enable_fallback = paddle.get_flags(enable_fallback_str)
    if enable_fallback:
        paddle.set_flags({enable_fallback_str: False})
    try:
        _run_static_single(place)
        _run_dygraph_single(place)
        print("TecoPaddle works well on {}.".format(place))
        print("paddle-sdaa and paddlepaddle are installed successfully!")

    except Exception as e:
        logging.warning(
            "Maybe some of SDAA kernels fallback to CPU,"
            " Check details on Original Error.\nOriginal Error is: {}".format(e)
        )

    finally:
        if enable_fallback:
            paddle.set_flags({enable_fallback_str: True})


def run_check():
    """
    Check whether paddle-sdaa is installed correctly and running successfully
    on your system.

    Examples:
        .. code-block:: python
            import paddle_sdaa
            paddle_sdaa.utils.run_check()

            #TecoPaddle works well on Place(sdaa:7).
            #paddle-sdaa and paddlepaddle are installed successfully!

    """
    _version_check()
    is_available, places = _is_sdaa_available()
    if is_available:
        place = places[-1]
        _test_on_one_device(place)
