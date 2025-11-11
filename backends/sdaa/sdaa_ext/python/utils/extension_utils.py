# BSD 3- Clause License Copyright (c) 2025, Tecorigin Co., Ltd. All rights
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

import atexit
import collections
import hashlib
import importlib.abc
import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
import sysconfig
import textwrap
import threading
import warnings
from contextlib import contextmanager
from importlib import machinery

from setuptools.command import bdist_egg

try:
    from subprocess import DEVNULL  # py3
except ImportError:
    DEVNULL = open(os.devnull, "wb")

from paddle.base import core
from paddle.base.framework import OpProtoHolder
from paddle.sysconfig import get_include, get_lib

logger = logging.getLogger("utils.cpp_extension")
logger.setLevel(logging.INFO)
formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s")
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

OS_NAME = sys.platform

CLANG_COMPILE_FLAGS = [
    "-fno-common",
    "-dynamic",
    "-DNDEBUG",
    "-g",
    "-fwrapv",
    "-O3",
    "-arch",
    "x86_64",
]
CLANG_LINK_FLAGS = [
    "-dynamiclib",
    "-undefined",
    "dynamic_lookup",
    "-arch",
    "x86_64",
]

TECOCC_LINK_FLAGS = [
    "-flto",
    "-fPIC",
    "-shared",
    "--sdaa-link",
    "-fuse-ld=lld",
    "-lm",
]

COMMON_TECOCC_FLAGS = ["-DPADDLE_WITH_CUSTOM_DEVICE"]

GCC_MINI_VERSION = (5, 4, 0)
# Give warning if using wrong compiler
WRONG_COMPILER_WARNING = """
                        *************************************
                        *  Compiler Compatibility WARNING   *
                        *************************************

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Found that your compiler ({user_compiler}) is not compatible with the compiler
built Paddle for this platform, which is {paddle_compiler} on {platform}. Please
use {paddle_compiler} to compile your custom op. Or you may compile Paddle from
source using {user_compiler}, and then also use it compile your custom op.

See https://www.paddlepaddle.org.cn/documentation/docs/zh/install/compile/fromsource.html
for help with compiling Paddle from source.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""
# Give warning if used compiler version is incompatible
ABI_INCOMPATIBILITY_WARNING = """
                            **********************************
                            *    ABI Compatibility WARNING   *
                            **********************************

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Found that your compiler ({user_compiler} == {version}) may be ABI-incompatible with pre-installed Paddle!
Please use compiler that is ABI-compatible with GCC >= 5.4 (Recommended 8.2).

See https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html for ABI Compatibility
information

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""

DEFAULT_OP_ATTR_NAMES = [
    core.op_proto_and_checker_maker.kOpRoleAttrName(),
    core.op_proto_and_checker_maker.kOpRoleVarAttrName(),
    core.op_proto_and_checker_maker.kOpNameScopeAttrName(),
    core.op_proto_and_checker_maker.kOpCreationCallstackAttrName(),
    core.op_proto_and_checker_maker.kOpDeviceAttrName(),
    core.op_proto_and_checker_maker.kOpWithQuantAttrName(),
]


@contextmanager
def bootstrap_context():
    """
    Context to manage how to write `__bootstrap__` code in .egg
    """
    origin_write_stub = bdist_egg.write_stub
    bdist_egg.write_stub = custom_write_stub
    yield

    bdist_egg.write_stub = origin_write_stub


def load_op_meta_info_and_register_op(lib_filename):
    core.load_op_meta_info_and_register_op(lib_filename)
    return OpProtoHolder.instance().update_op_proto()


def custom_write_stub(resource, pyfile):
    """
    Customized write_stub function to allow us to inject generated python
    api codes into egg python file.
    """
    _stub_template = textwrap.dedent(
        """
        {custom_api}

        import os
        import sys
        import types
        import paddle
        import importlib.abc
        import importlib.util

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        so_path = os.path.join(cur_dir, "{resource}")

        def __bootstrap__():
            assert os.path.exists(so_path)
            # load custom op shared library with abs path
            custom_ops = paddle.utils.cpp_extension.load_op_meta_info_and_register_op(so_path)

            if os.name == 'nt' or sys.platform.startswith('darwin'):
                # Cpp Extension only support Linux now
                mod = types.ModuleType(__name__)
            else:
                try:
                    spec = importlib.util.spec_from_file_location(__name__, so_path)
                    assert spec is not None
                    mod = importlib.util.module_from_spec(spec)
                    assert isinstance(spec.loader, importlib.abc.Loader)
                    spec.loader.exec_module(mod)
                except ImportError:
                    mod = types.ModuleType(__name__)

            for custom_op in custom_ops:
                setattr(mod, custom_op, eval(custom_op))

        __bootstrap__()

        """
    ).lstrip()

    # NOTE: To avoid importing .so file instead of python file because they have same name,
    # we rename .so shared library to another name, see EasyInstallCommand.
    filename, ext = os.path.splitext(resource)
    resource = filename + "_pd_" + ext

    api_content = []
    if CustomOpInfo.instance().empty():
        print("Received len(custom_op) =  0, using cpp extension only")
    else:
        # Parse registering op information
        _, op_info = CustomOpInfo.instance().last()
        so_path = op_info.so_path

        new_custom_ops = load_op_meta_info_and_register_op(so_path)
        for op_name in new_custom_ops:
            api_content.append(_custom_api_content(op_name))
        print(
            "Received len(custom_op) =  %d, using custom operator" % len(new_custom_ops)
        )

    with open(pyfile, "w") as f:
        f.write(
            _stub_template.format(
                resource=resource, custom_api="\n\n".join(api_content)
            )
        )


OpInfo = collections.namedtuple("OpInfo", ["so_name", "so_path"])


class CustomOpInfo:
    """
    A global Singleton map to record all compiled custom ops information.
    """

    @classmethod
    def instance(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        assert not hasattr(
            self.__class__, "_instance"
        ), "Please use `instance()` to get CustomOpInfo object!"
        # NOTE(Aurelius84): Use OrderedDict to save more order information
        self.op_info_map = collections.OrderedDict()

    def add(self, op_name, so_name, so_path=None):
        self.op_info_map[op_name] = OpInfo(so_name, so_path)

    def last(self):
        """
        Return the last inserted custom op info.
        """
        assert len(self.op_info_map) > 0
        return next(reversed(self.op_info_map.items()))

    def empty(self):
        if self.op_info_map:
            return False
        return True


VersionFields = collections.namedtuple(
    "VersionFields",
    [
        "sources",
        "extra_compile_args",
        "extra_link_args",
        "library_dirs",
        "runtime_library_dirs",
        "include_dirs",
        "define_macros",
        "undef_macros",
    ],
)


class VersionManager:
    def __init__(self, version_field):
        self.version_field = version_field
        self.version = self.hasher(version_field)

    def hasher(self, version_field):
        from paddle.utils import flatten

        md5 = hashlib.md5()
        for field in version_field._fields:
            elem = getattr(version_field, field)
            if not elem:
                continue
            if isinstance(elem, (list, tuple, dict)):
                flat_elem = flatten(elem)
                md5 = combine_hash(md5, tuple(flat_elem))
            else:
                raise RuntimeError(
                    "Support types with list, tuple and dict, but received {} with {}.".format(
                        type(elem), elem
                    )
                )

        return md5.hexdigest()

    @property
    def details(self):
        return self.version_field._asdict()


def combine_hash(md5, value):
    """
    Return new hash value.
    DO NOT use `hash()` because it doesn't generate stable value between different process.
    See https://stackoverflow.com/questions/27522626/hash-function-in-python-3-3-returns-different-results-between-sessions
    """
    md5.update(repr(value).encode())
    return md5


def clean_object_if_change_cflags(so_path, extension):
    """
    If already compiling source before, we should check whether cflags
    have changed and delete the built object to re-compile the source
    even though source file content keeps unchanaged.
    """

    def serialize(path, version_info):
        assert isinstance(version_info, dict)
        with open(path, "w") as f:
            f.write(json.dumps(version_info, indent=4, sort_keys=True))

    def deserialize(path):
        assert os.path.exists(path)
        with open(path, "r") as f:
            content = f.read()
            return json.loads(content)

    # version file
    VERSION_FILE = "version.txt"
    base_dir = os.path.dirname(so_path)
    so_name = os.path.basename(so_path)
    version_file = os.path.join(base_dir, VERSION_FILE)

    # version info
    args = [getattr(extension, field, None) for field in VersionFields._fields]
    version_field = VersionFields._make(args)
    versioner = VersionManager(version_field)

    if os.path.exists(so_path) and os.path.exists(version_file):
        old_version_info = deserialize(version_file)
        so_version = old_version_info.get(so_name, None)
        # delete shared library file if version is changed to re-compile it.
        if so_version is not None and so_version != versioner.version:
            log_v(
                "Re-Compiling {}, because specified cflags have been changed. New signature {} has been saved into {}.".format(
                    so_name, versioner.version, version_file
                )
            )
            os.remove(so_path)
            # update new version information
            new_version_info = versioner.details
            new_version_info[so_name] = versioner.version
            serialize(version_file, new_version_info)
    else:
        # If compile at first time, save compiling detail information for debug.
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        details = versioner.details
        details[so_name] = versioner.version
        serialize(version_file, details)


def prepare_unix_sdaaflags(cflags):
    """
    Prepare all necessary compiled flags for tecocc compiling SDAA C files.
    """
    cflags = COMMON_TECOCC_FLAGS + ["-O3", "-flto", "-fPIC"] + cflags
    return cflags


def add_std_without_repeat(cflags, compiler_type, use_std14=False):
    """
    Append -std=c++11/14 in cflags if without specific it before.
    """
    cpp_flag_prefix = "/std:" if compiler_type == "msvc" else "-std="
    if not any(cpp_flag_prefix in flag for flag in cflags):
        suffix = "c++14" if use_std14 else "c++11"
        cpp_flag = cpp_flag_prefix + suffix
        cflags.append(cpp_flag)


def _get_base_path():
    """
    Return installed base dir path.
    """
    import paddle

    return os.path.join(os.path.dirname(paddle.__file__), "base")


def _get_core_name():
    """
    Return pybind DSO module name.
    """
    ext_name = ".so"
    return "libpaddle" + ext_name


def _reset_so_rpath(so_path):
    """
    NOTE(Aurelius84): Runtime path of libpaddle.so is modified into `@loader_path/../libs`
    in setup.py.in. While loading custom op, `@loader_path` is the dirname of custom op
    instead of `paddle/base`. So we modify `@loader_path` from custom dylib into `@rpath`
    to ensure dynamic loader find it correctly.

    Moreover, we will add `-rpath site-packages/paddle/base` while linking the dylib so
    that we don't need to set `LD_LIBRARY_PATH` any more.
    """
    assert os.path.exists(so_path)


def _get_include_dirs_when_compiling(compile_dir):
    """
    Get all include directories when compiling the PaddlePaddle
    source code.
    """
    include_dirs_file = "includes.txt"
    path = os.path.abspath(compile_dir)
    include_dirs_file = os.path.join(path, include_dirs_file)
    assert os.path.isfile(include_dirs_file), f"File {include_dirs_file} does not exist"
    with open(include_dirs_file, "r") as f:
        include_dirs = [line.strip() for line in f.readlines() if line.strip()]

    extra_dirs = ["paddle/base/platform"]
    all_include_dirs = list(include_dirs)
    for extra_dir in extra_dirs:
        for include_dir in include_dirs:
            d = os.path.join(include_dir, extra_dir)
            if os.path.isdir(d):
                all_include_dirs.append(d)
    all_include_dirs.append(path)
    all_include_dirs.sort()
    return all_include_dirs


def normalize_extension_kwargs(kwargs, use_sdaa=False):
    """
    Normalize include_dirs, library_dir and other attributes in kwargs.
    """
    assert isinstance(kwargs, dict)
    compile_include_dirs = []
    # NOTE: the "_compile_dir" argument is not public to users. It is only
    # reserved for internal usage. We do not guarantee that this argument
    # is always valid in the future release versions.
    compile_dir = kwargs.get("_compile_dir", None)
    if compile_dir:
        compile_include_dirs = _get_include_dirs_when_compiling(compile_dir)

    # append necessary include dir path of paddle
    include_dirs = list(kwargs.get("include_dirs", []))
    include_dirs.extend(compile_include_dirs)
    include_dirs.extend(find_paddle_includes(use_sdaa))
    include_dirs.extend(find_python_includes())

    kwargs["include_dirs"] = include_dirs

    # append necessary lib path of paddle
    library_dirs = kwargs.get("library_dirs", [])
    library_dirs.extend(find_paddle_libraries(use_sdaa))
    kwargs["library_dirs"] = library_dirs

    # append compile flags and check settings of compiler
    extra_compile_args = kwargs.get("extra_compile_args", [])
    if isinstance(extra_compile_args, dict):
        for compiler in ["cxx", "nvcc", "tecocc"]:
            if compiler not in extra_compile_args:
                extra_compile_args[compiler] = []

    # ----------------------- Linux Platform ----------------------- #
    extra_link_args = kwargs.get("extra_link_args", [])
    # On Linux, GCC support '-l:xxx.so' to specify the library name
    # without `lib` prefix.
    extra_link_args.append(f"-l:{_get_core_name()}")

    if use_sdaa:
        for dir in library_dirs:
            if re.search("paddle_custom_device", dir):
                extra_link_args.append("-l:libpaddle-sdaa.so")
                break

    add_compile_flag(extra_compile_args, ["-w"])  # disable warning

    if use_sdaa:
        extra_link_args.extend(TECOCC_LINK_FLAGS)

    kwargs["extra_link_args"] = extra_link_args

    # add runtime library dirs
    runtime_library_dirs = kwargs.get("runtime_library_dirs", [])
    runtime_library_dirs.extend(find_paddle_libraries(use_sdaa))
    kwargs["runtime_library_dirs"] = runtime_library_dirs

    if compile_dir is None:
        # Add this compile option to isolate base headers
        add_compile_flag(extra_compile_args, ["-DPADDLE_WITH_CUSTOM_KERNEL"])
    kwargs["extra_compile_args"] = extra_compile_args

    kwargs["language"] = "c++"
    return kwargs


def find_sdaa_home():
    """
    Use heuristic method to find sdaa path
    """
    # step 1. find in $SDAA_HOME
    sdaa_home = os.environ.get("SDAA_HOME")

    # step 2.  find path by `which tecocc`
    if sdaa_home is None:
        try:
            with open(os.devnull, "w") as devnull:
                tecocc_path = (
                    subprocess.check_output(["which", "tecocc"], stderr=devnull)
                    .decode()
                    .rstrip("\r\n")
                )
                sdaa_home = os.path.dirname(os.path.dirname(tecocc_path))
        except:
            sdaa_home = "/opt/tecoai/"
    # step 3. check whether path is valid
    if (
        sdaa_home
        and not os.path.exists(sdaa_home)
        and core.is_compiled_with_custom_device("sdaa")
    ):
        sdaa_home = None

    return sdaa_home


def find_sdaa_includes():
    """
    Use heuristic method to find sdaa include path
    """
    sdaa_home = find_sdaa_home()
    if sdaa_home is None:
        raise ValueError(
            "Not found sdaa runtime, please use `export SDAA_HOME= XXX` to specific it."
        )
    return [os.path.join(sdaa_home, "include")]


def find_paddle_includes(use_sdaa=False):
    """
    Return Paddle necessary include dir path.
    """
    # pythonXX/site-packages/paddle/include
    paddle_include_dir = get_include()
    if use_sdaa:
        paddle_sdaa_include_dir = os.path.join(
            os.path.dirname(os.path.dirname(paddle_include_dir)),
            "paddle_sdaa",
            "include",
        )
    third_party_dir = os.path.join(paddle_include_dir, "third_party")
    include_dirs = [paddle_include_dir, third_party_dir]
    if use_sdaa and os.path.exists(paddle_sdaa_include_dir):
        include_dirs.append(paddle_sdaa_include_dir)

    if use_sdaa:
        sdaa_include_dir = find_sdaa_includes()
        include_dirs.extend(sdaa_include_dir)

    return include_dirs


def find_python_includes():
    """
    Return necessary include dir path of Python.h.
    """
    # sysconfig.get_path('include') gives us the location of Python.h
    # Explicitly specify 'posix_prefix' scheme on non-Windows platforms to workaround error on some MacOS
    # installations where default `get_path` points to non-existing `/Library/Python/M.m/include` folder
    python_include_path = sysconfig.get_path("include", scheme="posix_prefix")
    if python_include_path is not None:
        assert isinstance(python_include_path, str)
        return [python_include_path]
    return []


def find_sdaa_libraries():
    """
    Use heuristic method to find sdaa dynamic lib path
    """
    sdaa_home = find_sdaa_home()
    if sdaa_home is None:
        raise ValueError(
            "Not found SDAA runtime, please use `export SDAA_HOME=XXX` to specific it."
        )
    sdaa_lib_dir = [os.path.join(sdaa_home, "lib64")]

    return sdaa_lib_dir


def find_paddle_libraries(use_sdaa=False):
    """
    Return Paddle necessary library dir path.
    """
    # pythonXX/site-packages/paddle/libs
    paddle_lib_dirs = [get_lib()]
    if use_sdaa:
        paddle_sdaa_lib_dirs = os.path.join(
            os.path.dirname(os.path.dirname(paddle_lib_dirs[0])),
            "paddle_custom_device",
        )
        if os.path.exists(paddle_sdaa_lib_dirs):
            paddle_lib_dirs.append(paddle_sdaa_lib_dirs)

    if use_sdaa:
        sdaa_lib_dir = find_sdaa_libraries()
        paddle_lib_dirs.extend(sdaa_lib_dir)

    # add `paddle/base` to search `libpaddle.so`
    paddle_lib_dirs.append(_get_base_path())

    return paddle_lib_dirs


def add_compile_flag(extra_compile_args, flags):
    assert isinstance(flags, list)
    if isinstance(extra_compile_args, dict):
        for args in extra_compile_args.values():
            args.extend(flags)
    else:
        extra_compile_args.extend(flags)


def is_sdaa_file(path):
    sdaa_suffix = {".scpp"}
    items = os.path.splitext(path)
    assert len(items) > 1
    return items[-1] in sdaa_suffix


def get_build_directory(verbose=False):
    """
    Return paddle extension root directory to put shared library. It could be specified by
    ``export PADDLE_EXTENSION_DIR=XXX`` . If not set, ``~/.cache/paddle_extension`` will be used
    by default.

    Returns:
        The root directory of compiling customized operators.

    Examples:

    .. code-block:: python

        >>> from paddle.utils.cpp_extension import get_build_directory

        >>> build_dir = get_build_directory()
        >>> print(build_dir)

    """
    root_extensions_directory = os.environ.get("PADDLE_EXTENSION_DIR")
    if root_extensions_directory is None:
        dir_name = "paddle_extensions"
        root_extensions_directory = os.path.join(
            os.path.expanduser("~/.cache"), dir_name
        )

        log_v(
            "$PADDLE_EXTENSION_DIR is not set, using path: {} by default.".format(
                root_extensions_directory
            ),
            verbose,
        )

    if not os.path.exists(root_extensions_directory):
        os.makedirs(root_extensions_directory)

    return root_extensions_directory


def parse_op_info(op_name):
    """
    Parse input names and outpus detail information from registered custom op
    from OpInfoMap.
    """
    if op_name not in OpProtoHolder.instance().op_proto_map:
        raise ValueError(
            f"Please load {op_name} shared library file firstly by `paddle.utils.cpp_extension.load_op_meta_info_and_register_op(...)`"
        )
    op_proto = OpProtoHolder.instance().get_op_proto(op_name)

    in_names = [x.name for x in op_proto.inputs]
    attr_names = [x.name for x in op_proto.attrs if x.name not in DEFAULT_OP_ATTR_NAMES]
    out_names = [x.name for x in op_proto.outputs]

    return in_names, attr_names, out_names


def _import_module_from_library(module_name, build_directory, verbose=False):
    """
    Load shared library and import it as callable python module.
    """
    dynamic_suffix = ".so"
    ext_path = os.path.join(build_directory, module_name + dynamic_suffix)
    if not os.path.exists(ext_path):
        raise FileNotFoundError(f"Extension path: {ext_path} does not exist.")

    # load custom op_info and kernels from .so shared library
    log_v(f"loading shared library from: {ext_path}", verbose)
    op_names = load_op_meta_info_and_register_op(ext_path)

    try:
        spec = importlib.util.spec_from_file_location(module_name, ext_path)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        assert isinstance(spec.loader, importlib.abc.Loader)
        spec.loader.exec_module(module)
    except ImportError:
        log_v("using custom operator only")
        return _generate_python_module(module_name, op_names, build_directory, verbose)

    # generate Python api in ext_path
    op_module = _generate_python_module(module_name, op_names, build_directory, verbose)
    for op_name in op_names:
        # Mix use of Cpp Extension and Custom Operator
        setattr(module, op_name, getattr(op_module, op_name))

    return module


def _generate_python_module(module_name, op_names, build_directory, verbose=False):
    """
    Automatically generate python file to allow import or load into as module
    """

    def remove_if_exit(filepath):
        if os.path.exists(filepath):
            os.remove(filepath)

    # NOTE: Use unique id as suffix to avoid write same file at same time in
    # both multi-thread and multi-process.
    thread_id = str(threading.currentThread().ident)
    api_file = os.path.join(build_directory, module_name + "_" + thread_id + ".py")
    log_v(f"generate api file: {api_file}", verbose)

    # delete the temp file before exit python process
    atexit.register(lambda: remove_if_exit(api_file))

    # write into .py file with RWLockc
    api_content = [_custom_api_content(op_name) for op_name in op_names]
    with open(api_file, "w") as f:
        f.write("\n\n".join(api_content))

    # load module
    custom_module = _load_module_from_file(api_file, module_name, verbose)
    return custom_module


def _gen_output_content(
    op_name,
    in_names,
    out_names,
    ins_map,
    attrs_map,
    outs_list,
    inplace_reverse_idx,
):
    # ' ' * tab space * tab number
    indent = " " * 4 * 2
    dynamic_content = f"""res = []
{indent}start_idx = 0"""
    static_content = f"""ins = {{}}
{indent}ins_map = {ins_map}
{indent}outs = {{}}
{indent}outs_list = {outs_list}
{indent}for key, value in ins_map.items():
{indent}    # handle optional inputs
{indent}    if value is not None:
{indent}        ins[key] = value
{indent}helper = LayerHelper("{op_name}", **locals())
"""
    for out_idx, out_name in enumerate(out_names):
        in_idx = -1
        if out_idx in inplace_reverse_idx:
            in_idx = inplace_reverse_idx[out_idx]
        if (
            in_idx != -1
            and "@VECTOR" in in_names[in_idx]
            and "@OPTIONAL" in in_names[in_idx]
        ):
            # inplace optional vector<Tensor> output case
            lower_in_names = in_names[in_idx].split("@")[0].lower()
            dynamic_content += f"""
{indent}if {lower_in_names} is not None:
{indent}    res.append(outs[start_idx: start_idx + len({lower_in_names})])
{indent}    start_idx += len({lower_in_names})
{indent}else:
{indent}    res.append(None)
{indent}    start_idx += 1"""
            static_content += f"""
{indent}if {lower_in_names} is not None:
{indent}    outs['{out_name}'] = {lower_in_names}"""

        elif (
            in_idx != -1 and "@VECTOR" in in_names[in_idx]
        ):  # inplace vector<Tensor> output case
            lower_in_names = in_names[in_idx].split("@")[0].lower()
            dynamic_content += f"""
{indent}res.append(outs[start_idx: start_idx + len({lower_in_names})])
{indent}start_idx += len({lower_in_names})"""
            static_content += f"""
{indent}outs['{out_name}'] = {lower_in_names}"""
        elif (
            in_idx != -1 and "@OPTIONAL" in in_names[in_idx]
        ):  # inplace optional Tensor output case, handle inplace None input
            lower_in_names = in_names[in_idx].split("@")[0].lower()
            dynamic_content += f"""
{indent}if {lower_in_names} is not None:
{indent}    res.append(outs[start_idx])
{indent}else:
{indent}    res.append(None)
{indent}start_idx += 1"""
            static_content += f"""
{indent}if {lower_in_names} is not None:
{indent}    outs['{out_name}'] = {lower_in_names}"""
        elif in_idx != -1:  # inplace Tensor output case, handle inplace None input
            lower_in_names = in_names[in_idx].lower()
            dynamic_content += f"""
{indent}res.append(outs[start_idx])
{indent}start_idx += 1"""
            static_content += f"""
{indent}outs['{out_name}'] = {lower_in_names}"""
        else:  # general/inplace Tensor output case
            dynamic_content += f"""
{indent}res.append(outs[start_idx])
{indent}start_idx += 1"""
            static_content += f"""
{indent}outs['{out_name}'] = helper.create_variable(dtype='float32')"""

    dynamic_content += f"""
{indent}return res[0] if len(res)==1 else res"""

    static_content += f"""
{indent}helper.append_op(type="{op_name}", inputs=ins, outputs=outs, attrs={attrs_map})
{indent}res = [outs[out_name] if out_name in outs.keys() else None for out_name in outs_list]
{indent}return res[0] if len(res)==1 else res"""

    return dynamic_content, static_content


def _custom_api_content(op_name):
    (
        params_list,
        ins_map,
        attrs_map,
        outs_list,
        in_names,
        attr_names,
        out_names,
        inplace_reverse_idx,
    ) = _get_api_inputs_str(op_name)
    dynamic_content, static_content = _gen_output_content(
        op_name,
        in_names,
        out_names,
        ins_map,
        attrs_map,
        outs_list,
        inplace_reverse_idx,
    )
    API_TEMPLATE = textwrap.dedent(
        """
        import paddle.base.core as core
        from paddle.framework import in_dynamic_mode
        from paddle.base.layer_helper import LayerHelper

        def {op_name}({params_list}):
            # The output variable's dtype use default value 'float32',
            # and the actual dtype of output variable will be inferred in runtime.
            if in_dynamic_mode():
                outs = core.eager._run_custom_op("{op_name}", {params_list})
                {dynamic_content}
            else:
                {static_content}
            """
    ).lstrip()

    # generate python api file
    api_content = API_TEMPLATE.format(
        op_name=op_name,
        params_list=params_list,
        dynamic_content=dynamic_content,
        static_content=static_content,
    )

    return api_content


def _load_module_from_file(api_file_path, module_name, verbose=False):
    """
    Load module from python file.
    """
    if not os.path.exists(api_file_path):
        raise FileNotFoundError(f"File : {api_file_path} does not exist.")

    # Unique readable module name to place custom api.
    log_v(f"import module from file: {api_file_path}", verbose)
    ext_name = "_paddle_cpp_extension_" + module_name

    # load module with RWLock
    loader = machinery.SourceFileLoader(ext_name, api_file_path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)

    return module


def _get_api_inputs_str(op_name):
    """
    Returns string of api parameters and inputs dict.
    """
    in_names, attr_names, out_names = parse_op_info(op_name)
    # e.g: x, y, z
    param_names = in_names + attr_names
    # NOTE(chenweihang): we add suffix `@VECTOR` for std::vector<Tensor> input,
    # but the string contains `@` cannot used as argument name, so we split
    # input name by `@`, and only use first substr as argument
    params_list = ",".join([p.split("@")[0].lower() for p in param_names])
    # e.g: {'X': x, 'Y': y, 'Z': z}
    ins_map = "{%s}" % ",".join(
        [
            "'{}' : {}".format(in_name, in_name.split("@")[0].lower())
            for in_name in in_names
        ]
    )
    # e.g: {'num': n}
    attrs_map = "{%s}" % ",".join(
        [
            "'{}' : {}".format(attr_name, attr_name.split("@")[0].lower())
            for attr_name in attr_names
        ]
    )
    # e.g: ['Out', 'Index']
    outs_list = "[%s]" % ",".join([f"'{name}'" for name in out_names])

    inplace_reverse_idx = core.eager._get_custom_operator_inplace_map(op_name)

    return (
        params_list,
        ins_map,
        attrs_map,
        outs_list,
        in_names,
        attr_names,
        out_names,
        inplace_reverse_idx,
    )


def _write_setup_file(
    name,
    sources,
    file_path,
    build_dir,
    include_dirs,
    library_dirs,
    extra_cxx_cflags,
    extra_sdaa_cflags,
    link_args,
    verbose=False,
):
    """
    Automatically generate setup.py and write it into build directory.
    """
    template = textwrap.dedent(
        """
    import os
    from paddle.utils.cpp_extension import CppExtension
    from paddle_sdaa.utils.sdaa_extension import SDAAExtension, BuildExtension, setup
    from paddle_sdaa.utils.sdaa_extension import get_build_directory


    setup(
        name='{name}',
        ext_modules=[
            {prefix}Extension(
                sources={sources},
                include_dirs={include_dirs},
                library_dirs={library_dirs},
                extra_compile_args={{'cxx':{extra_cxx_cflags}, 'tecocc':{extra_sdaa_cflags}}},
                extra_link_args={extra_link_args})],
        cmdclass={{"build_ext" : BuildExtension.with_options(
            output_dir=r'{build_dir}',
            no_python_abi_suffix=True)
        }})"""
    ).lstrip()

    with_sdaa = False
    if any(is_sdaa_file(source) for source in sources):
        with_sdaa = True
    log_v(f"with_sdaa: {with_sdaa}", verbose)

    prefix = "Cpp"
    if with_sdaa:
        prefix = "SDAA"

    content = template.format(
        name=name,
        prefix=prefix,
        sources=list2str(sources),
        include_dirs=list2str(include_dirs),
        library_dirs=list2str(library_dirs),
        extra_cxx_cflags=list2str(extra_cxx_cflags),
        extra_sdaa_cflags=list2str(extra_sdaa_cflags),
        extra_link_args=list2str(link_args),
        build_dir=build_dir,
    )

    log_v(f"write setup.py into {file_path}", verbose)
    with open(file_path, "w") as f:
        f.write(content)


def list2str(args):
    """
    Convert list[str] into string. For example: ['x', 'y'] -> "['x', 'y']"
    """
    if args is None:
        return "[]"
    assert isinstance(args, (list, tuple))
    args = [f"{arg}" for arg in args]
    return repr(args)


def _jit_compile(file_path, verbose=False):
    """
    Build shared library in subprocess
    """
    ext_dir = os.path.dirname(file_path)
    setup_file = os.path.basename(file_path)

    # Using interpreter same with current process.
    interpreter = sys.executable

    try:
        py_version = subprocess.check_output([interpreter, "-V"])
        py_version = py_version.decode()
        log_v(
            f"Using Python interpreter: {interpreter}, version: {py_version.strip()}",
            verbose,
        )
    except Exception:
        _, error, _ = sys.exc_info()
        raise RuntimeError(
            f"Failed to check Python interpreter with `{interpreter}`, errors: {error}"
        )

    compile_cmd = f"cd {ext_dir} && {interpreter} {setup_file} build"

    print("Compiling user custom op, it will cost a few seconds.....")
    run_cmd(compile_cmd, verbose)


def parse_op_name_from(sources):
    """
    Parse registerring custom op name from sources.
    """

    def regex(content):
        pattern = re.compile(r"PD_BUILD_OP\(([^,\)]+)\)")
        content = re.sub(r"\s|\t|\n", "", content)
        op_name = pattern.findall(content)
        op_name = {re.sub("_grad", "", name) for name in op_name}

        return op_name

    op_names = set()
    for source in sources:
        with open(source, "r", encoding="utf-8") as f:
            content = f.read()
            op_names |= regex(content)

    return list(op_names)


def run_cmd(command, verbose=False):
    """
    Execute command with subprocess.
    """
    # logging
    log_v(f"execute command: {command}", verbose)

    # execute command
    try:
        if verbose:
            return subprocess.check_call(command, shell=True, stderr=subprocess.STDOUT)
        else:
            return subprocess.check_call(command, shell=True, stdout=DEVNULL)
    except Exception:
        _, error, _ = sys.exc_info()
        raise RuntimeError(f"Failed to run command: {compile}, errors: {error}")


def check_abi_compatibility(compiler, verbose=False):
    """
    Check whether GCC version on user local machine is compatible with Paddle in
    site-packages.
    """
    if os.environ.get("PADDLE_SKIP_CHECK_ABI") in ["True", "true", "1"]:
        return True

    cmd_out = subprocess.check_output(["which", compiler], stderr=subprocess.STDOUT)
    compiler_path = os.path.realpath(cmd_out.decode()).strip()
    # if not found any suitable compiler, raise warning
    if not any(name in compiler_path for name in _expected_compiler_current_platform()):
        warnings.warn(
            WRONG_COMPILER_WARNING.format(
                user_compiler=compiler,
                paddle_compiler=_expected_compiler_current_platform()[0],
                platform=OS_NAME,
            )
        )
        return False

    version = (0, 0, 0)
    try:
        mini_required_version = GCC_MINI_VERSION
        version_info = subprocess.check_output(
            [compiler, "-dumpfullversion", "-dumpversion"]
        )
        version_info = version_info.decode()
        version = version_info.strip().split(".")
    except Exception:
        # check compiler version failed
        _, error, _ = sys.exc_info()
        warnings.warn(f"Failed to check compiler version for {compiler}: {error}")
        return False

    # check version compatibility
    assert len(version) == 3
    if tuple(map(int, version)) >= mini_required_version:
        return True
    warnings.warn(
        ABI_INCOMPATIBILITY_WARNING.format(
            user_compiler=compiler, version=".".join(version)
        )
    )
    return False


def _expected_compiler_current_platform():
    """
    Returns supported compiler string on current platform
    """
    expect_compilers = ["gcc", "g++", "gnu-c++", "gnu-cc"]
    return expect_compilers


def log_v(info, verbose=True):
    """
    Print log information on stdout.
    """
    if verbose:
        logger.info(info)
