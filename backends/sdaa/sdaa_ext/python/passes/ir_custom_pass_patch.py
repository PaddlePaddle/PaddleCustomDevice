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
from paddle.incubate.passes import ir
from types import MethodType
from paddle.base.libpaddle import OpDesc
import logging

LOG_FORMAT = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.basicConfig(format=LOG_FORMAT)

_PRUNE_OP_MAP = {"batch_norm": {"Outputs": {"ReserveSpace"}}}


def _prune_op(op: OpDesc):
    op_type = op.type()
    removed_key = []
    if op_type in _PRUNE_OP_MAP:
        for name in _PRUNE_OP_MAP[op_type].get("Inputs", []):
            op.remove_input(name)
            removed_key.append(name)

        for name in _PRUNE_OP_MAP[op_type].get("Outputs", []):
            op.remove_output(name)
            removed_key.append(name)

        for name in _PRUNE_OP_MAP[op_type].get("Attrs", []):
            op.remove_attr(name)
            removed_key.append(name)
    if removed_key:
        logger.warning(f"custom ir pass pruned {op_type}: {removed_key}")


class PyMethodWrapper:
    def __init__(self, func):
        self._func = func

    def __get__(self, instance, cls=None):
        return MethodType(self, instance) if instance else self

    def __call__(self, instance, func, ops):
        vars = []
        program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(program, startup_program):
            args = instance._get_args_from_func(func)
            vars.extend(args)
            outs = func(*args)
            if not isinstance(outs, (list, tuple)):
                outs = [outs]
            for out in outs:
                if isinstance(out, ir.PassDesc.OpHelper):
                    op_outs = out.Outputs()
                    if len(op_outs) != 1:
                        raise ValueError(
                            f"Operator '{out._type}' has multiple outputs"
                            ", please specify one output variable."
                        )
                    for op_out in op_outs.values():
                        vars.extend(op_out)
                else:
                    vars.append(out)
        block_desc = program.current_block().desc
        for i in range(block_desc.op_size()):
            _prune_op(block_desc.op(i))
            ops.add().ParseFromString(block_desc.op(i).serialize_to_string())

        instance._prune_program_desc(ops)

        return vars, program.current_block().ops


def monkey_patch_for_custom_pass():
    if paddle.__version__ > "2.6.0":
        raise Exception(
            f"not sure custom pass monkey path is valid for {paddle.__version__}, please check it."
        )

    new_func_to_program_desc = PyMethodWrapper(
        ir.RegisterPassHelper._func_to_program_desc
    )

    ir.RegisterPassHelper._func_to_program_desc = new_func_to_program_desc
