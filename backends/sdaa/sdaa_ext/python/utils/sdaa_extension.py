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

import os
import copy

import setuptools
from setuptools.command.easy_install import easy_install
from setuptools.command.build_ext import build_ext
from distutils.command.build import build
from distutils.dep_util import newer_group
from distutils.errors import DistutilsSetupError
from distutils import log

from .extension_utils import (
    add_compile_flag,
    find_sdaa_home,
    normalize_extension_kwargs,
)
from .extension_utils import (
    is_sdaa_file,
    prepare_unix_sdaaflags,
)
from .extension_utils import (
    _import_module_from_library,
    _write_setup_file,
    _jit_compile,
)
from .extension_utils import (
    check_abi_compatibility,
    log_v,
    CustomOpInfo,
    parse_op_name_from,
)
from .extension_utils import _reset_so_rpath, clean_object_if_change_cflags
from .extension_utils import (
    bootstrap_context,
    get_build_directory,
    add_std_without_repeat,
)

SDAA_HOME = find_sdaa_home()

CC_LINK_FLAGS = ["-fPIC", "-shared"]


def setup(**attr):
    """
    The interface is used to config the process of compiling customized operators,
    mainly includes how to compile shared library, automatically generate python API
    and install it into site-package. It supports using customized operators directly with
    ``import`` statement.

    It encapsulates the python built-in ``setuptools.setup`` function and keeps arguments
    and usage same as the native interface. Meanwhile, it hides Paddle inner framework
    concepts, such as necessary compiling flags, included paths of head files, and linking
    flags. It also will automatically search and valid local environment and versions of
    ``cc(Linux)`` , then compiles customized operators
    supporting CPU or GPU device according to the specified Extension type.

    Moreover, `ABI compatibility <https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html>`_
    will be checked to ensure that compiler version from ``cc(Linux)`` , ``cl.exe(Windows)``
    on local machine is compatible with pre-installed Paddle whl in python site-packages.

    For Linux, GCC version will be checked . For example if Paddle with CUDA 10.1 is built with GCC 8.2,
    then the version of user's local machine should satisfy GCC >= 8.2.
    For Windows, Visual Studio version will be checked, and it should be greater than or equal to that of
    PaddlePaddle (Visual Studio 2017).
    If the above conditions are not met, the corresponding warning will be printed, and a fatal error may
    occur because of ABI compatibility.

    Note:

        1. Currently we support Linux, MacOS and Windows platform.
        2. On Linux platform, we recommend to use GCC 8.2 as soft linking candidate of ``/usr/bin/cc`` .
           Then, Use ``which cc`` to ensure location of ``cc`` and using ``cc --version`` to ensure linking
           GCC version.
        3. On Windows platform, we recommend to install `` Visual Studio`` (>=2017).


    Compared with Just-In-Time ``load`` interface, it only compiles once by executing
    ``python setup.py install`` . Then customized operators API will be available everywhere
    after importing it.

    A simple example of ``setup.py`` as followed:

    .. code-block:: text

        # setup.py

        # Case 1: Compiling customized operators supporting CPU and GPU devices
        from paddle.utils.cpp_extension import CUDAExtension, setup

        setup(
            name='custom_op',  # name of package used by "import"
            ext_modules=CUDAExtension(
                sources=['relu_op.cc', 'relu_op.cu', 'tanh_op.cc', 'tanh_op.cu']  # Support for compilation of multiple OPs
            )
        )

        # Case 2: Compiling customized operators supporting only CPU device
        from paddle.utils.cpp_extension import CppExtension, setup

        setup(
            name='custom_op',  # name of package used by "import"
            ext_modules=CppExtension(
                sources=['relu_op.cc', 'tanh_op.cc']  # Support for compilation of multiple OPs
            )
        )


    Applying compilation and installation by executing ``python setup.py install`` under source files directory.
    Then we can use the layer api as followed:

    .. code-block:: text

        import paddle
        from custom_op import relu, tanh

        x = paddle.randn([4, 10], dtype='float32')
        relu_out = relu(x)
        tanh_out = tanh(x)


    Args:
        name(str): Specify the name of shared library file and installed python package.
        ext_modules(Extension): Specify the Extension instance including customized operator source files, compiling flags et.al.
                                If only compile operator supporting CPU device, please use ``CppExtension`` ; If compile operator
                                supporting CPU and GPU devices, please use ``CUDAExtension`` .
        include_dirs(list[str], optional): Specify the extra include directories to search head files. The interface will automatically add
                                 ``site-package/paddle/include`` . Please add the corresponding directory path if including third-party
                                 head files. Default is None.
        extra_compile_args(list[str] | dict, optional): Specify the extra compiling flags such as ``-O3`` . If set ``list[str]`` , all these flags
                                will be applied for ``cc`` and ``nvcc`` compiler. It supports specify flags only applied ``cc`` or ``nvcc``
                                compiler using dict type with ``{'cxx': [...], 'nvcc': [...]}`` . Default is None.
        **attr(dict, optional): Specify other arguments same as ``setuptools.setup`` .

    Returns:
        None

    """
    cmdclass = attr.get("cmdclass", {})
    assert isinstance(cmdclass, dict)
    # if not specific cmdclass in setup, add it automatically.
    if "build_ext" not in cmdclass:
        cmdclass["build_ext"] = BuildExtension.with_options(no_python_abi_suffix=True)
        attr["cmdclass"] = cmdclass

    error_msg = """
    Required to specific `name` argument in paddle_sdaa.utils.sdaa_extension.setup.
    It's used as `import XXX` when you want install and import your custom operators.\n
    For Example:
        # setup.py file
        from paddle_sdaa.utils.sdaa_extension import SDAAExtension, setup
        setup(name='custom_module',
              ext_modules=SDAAExtension(
              sources=['relu_op.cc', 'relu_op.scpp'])

        # After running `python setup.py install`
        from custom_module import relu
    """
    # name argument is required
    if "name" not in attr:
        raise ValueError(error_msg)

    assert not attr["name"].endswith(
        "module"
    ), "Please don't use 'module' as suffix in `name` argument, "
    "it will be stripped in setuptools.bdist_egg and cause import error."

    ext_modules = attr.get("ext_modules", [])
    if not isinstance(ext_modules, list):
        ext_modules = [ext_modules]
    assert (
        len(ext_modules) == 1
    ), "Required only one Extension, but received {}. If you want to compile multi operators, you can include all necessary source files in one Extension.".format(
        len(ext_modules)
    )
    # replace Extension.name with attr['name] to keep consistent with Package name.
    for ext_module in ext_modules:
        ext_module.name = attr["name"]

    attr["ext_modules"] = ext_modules

    # Add rename .so hook in easy_install
    assert "easy_install" not in cmdclass
    cmdclass["easy_install"] = EasyInstallCommand

    # Note(Aurelius84): Add rename build_base directory hook in build command.
    # To avoid using same build directory that will lead to remove the directory
    # by mistake while parallelling execute setup.py, for example on CI.
    assert "build" not in cmdclass
    build_base = os.path.join("build", attr["name"])
    cmdclass["build"] = BuildCommand.with_options(build_base=build_base)

    # Always set zip_safe=False to make compatible in PY2 and PY3
    # See http://peak.telecommunity.com/DevCenter/setuptools#setting-the-zip-safe-flag
    attr["zip_safe"] = False

    # switch `write_stub` to inject paddle api in .egg
    with bootstrap_context():
        setuptools.setup(**attr)


def SDAAExtension(sources, *args, **kwargs):
    """
    The interface is used to config source files of customized operators and complies
    Op Kernel supporting both CPU and SDAA devices. Please use ``CppExtension`` if you want to
    compile Op Kernel that supports only CPU device.

    It further encapsulates python built-in ``setuptools.Extension`` .The arguments and
    usage are same as the native interface, except for no need to explicitly specify
    ``name`` .

    **A simple example:**

    .. code-block:: text

        # setup.py

        # Compiling customized operators supporting CPU and SDAA devices
        from paddle_sdaa.utils.sdaa_extension import SDAAExtension, setup

        setup(
            name='custom_op',
            ext_modules=SDAAExtension(
                sources=['relu_op.cc', 'relu_op.scpp']
            )
        )


    Note:
        It is mainly used in ``setup`` and the name of built shared library keeps same
        as ``name`` argument specified in ``setup`` interface.


    Args:
        sources(list[str]): Specify the C++/SDAA C source files of customized operators.
        *args(list[options], optional): Specify other arguments same as ``setuptools.Extension`` .
        **kwargs(dict[option], optional): Specify other arguments same as ``setuptools.Extension`` .

    Returns:
        setuptools.Extension: An instance of setuptools.Extension.
    """
    kwargs = normalize_extension_kwargs(kwargs, use_sdaa=True)
    name = kwargs.get("name", None)
    if name is None:
        name = _generate_extension_name(sources)

    return setuptools.Extension(name, sources, *args, **kwargs)


def _generate_extension_name(sources):
    """
    Generate extension name by source files.
    """
    assert len(sources) > 0, "source files is empty"
    file_prefix = []
    for source in sources:
        source = os.path.basename(source)
        filename, _ = os.path.splitext(source)
        # Use list to generate same order.
        if filename not in file_prefix:
            file_prefix.append(filename)

    return "_".join(file_prefix)


class BuildExtension(build_ext):
    """
    Inherited from setuptools.command.build_ext to customize how to apply
    compilation process with share library.
    """

    @classmethod
    def with_options(cls, **options):
        """
        Returns a BuildExtension subclass containing use-defined options.
        """

        class cls_with_options(cls):
            def __init__(self, *args, **kwargs):
                kwargs.update(options)
                cls.__init__(self, *args, **kwargs)

        return cls_with_options

    def __init__(self, *args, **kwargs):
        """
        Attributes is initialized with following order:

            1. super().__init__()
            2. initialize_options(self)

            3. the reset of current __init__()
            4. finalize_options(self)

        So, it is recommended to set attribute value in `finalize_options`.
        """
        super().__init__(*args, **kwargs)
        self.no_python_abi_suffix = kwargs.get("no_python_abi_suffix", True)
        self.output_dir = kwargs.get("output_dir", None)
        # whether containing sdaa source file in Extensions
        self.contain_sdaa_file = False

    def initialize_options(self):
        super().initialize_options()

    def finalize_options(self):
        super().finalize_options()
        # NOTE(Aurelius84): Set location of compiled shared library.
        # Carefully to modify this because `setup.py build/install`
        # and `load` interface rely on this attribute.
        if self.output_dir is not None:
            self.build_lib = self.output_dir

    def sdaa_build_extension(self, ext):
        sources = ext.sources
        if sources is None or not isinstance(sources, (list, tuple)):
            raise DistutilsSetupError(
                "in 'ext_modules' option (extension '%s'), "
                "'sources' must be present and must be "
                "a list of source filenames" % ext.name
            )
        # sort to make the resulting .so file build reproducible
        sources = sorted(sources)

        ext_path = self.get_ext_fullpath(ext.name)
        depends = sources + ext.depends
        if not (self.force or newer_group(depends, ext_path, "newer")):
            log.debug("skipping '%s' extension (up-to-date)", ext.name)
            return
        else:
            log.info("building '%s' extension", ext.name)

        # First, scan the sources for SWIG definition files (.i), run
        # SWIG on 'em to create .c files, and modify the sources list
        # accordingly.
        sources = self.swig_sources(sources, ext)
        with_sdaa = any(is_sdaa_file(source) for source in sources)

        # XXX not honouring 'define_macros' or 'undef_macros' -- the
        # CCompiler API needs to change to accommodate this, and I
        # want to do one thing at a time!
        macros = ext.define_macros[:]
        for undef in ext.undef_macros:
            macros.append((undef,))

        # Two possible sources for extra compiler arguments:
        #   - 'extra_compile_args' in Extension object
        #   - CFLAGS environment variable (not particularly
        #     elegant, but people seem to expect it and I
        #     guess it's useful)
        # The environment variable should take precedence, and
        # any sensible compiler will give precedence to later
        # command line args.  Hence we combine them in order:
        compile_extra_args = ext.extra_compile_args or {"cxx": [], "tecocc": []}
        link_extra_args = ext.extra_link_args or {"cxx": [], "tecocc": []}
        all_libraries = copy.deepcopy(ext.libraries) or {
            "cxx": [],
            "tecocc": [],
        }

        def collect_all_sdaa_file(sources):
            sdaa_files = []
            for source in sources:
                if is_sdaa_file(source):
                    sdaa_files.append(source)
            return sdaa_files

        def collect_all_cxx_file(sources):
            cxx_files = []
            for source in sources:
                if not is_sdaa_file(source):
                    cxx_files.append(source)
            return cxx_files

        def unix_single_compile(sources):
            language = ext.language or self.compiler.detect_language(sources)
            # Process1. collect all .scpp file, compile and link use tecocc
            if with_sdaa:
                sdaa_files = collect_all_sdaa_file(sources)

                original_compiler_so = self.compiler.compiler_so
                original_compiler_cxx = self.compiler.compiler_cxx
                original_linker_so = self.compiler.linker_so

                tecocc_cmd = os.path.join(SDAA_HOME, "bin", "tecocc")
                self.compiler.set_executable("compiler_so", tecocc_cmd)
                self.compiler.set_executable("compiler_cxx", tecocc_cmd)
                self.compiler.set_executable("linker_so", tecocc_cmd)

                ext.libraries = (
                    all_libraries["tecocc"]
                    if isinstance(all_libraries, dict)
                    else all_libraries
                )
                sdaa_objects = self.compiler.compile(
                    sdaa_files,
                    output_dir=self.build_temp,
                    macros=macros,
                    include_dirs=ext.include_dirs,
                    debug=self.debug,
                    extra_postargs=compile_extra_args["tecocc"]
                    if isinstance(compile_extra_args, dict)
                    else compile_extra_args,
                    depends=ext.depends,
                )

                sdaa_so_name = f"libteco_{ext.name}.so"
                self.compiler.link_shared_object(
                    sdaa_objects,
                    sdaa_so_name,  # output_filename
                    output_dir=self.build_temp,
                    libraries=self.get_libraries(ext),
                    library_dirs=ext.library_dirs,
                    runtime_library_dirs=ext.runtime_library_dirs,
                    extra_postargs=link_extra_args["tecocc"]
                    if isinstance(link_extra_args, dict)
                    else link_extra_args,
                    export_symbols=self.get_export_symbols(ext),
                    debug=self.debug,
                    build_temp=self.build_temp,
                    target_lang=language,
                )

                self.compiler.set_executable("compiler_so", original_compiler_so)
                self.compiler.set_executable("compiler_cxx", original_compiler_cxx)
                self.compiler.set_executable("linker_so", original_linker_so)

            # Process 2. collect all cpp file, compile and link all object
            c_files = collect_all_cxx_file(sources)
            c_objects = self.compiler.compile(
                c_files,
                output_dir=self.build_temp,
                macros=macros,
                include_dirs=ext.include_dirs,
                debug=self.debug,
                extra_postargs=compile_extra_args["cxx"]
                if isinstance(compile_extra_args, dict)
                else compile_extra_args,
                depends=ext.depends,
            )

            # XXX outdated variable, kept here in case third-part code needs it.
            self._built_objects = sdaa_objects + c_objects

            ext.libraries = all_libraries["cxx"]
            libraries = self.get_libraries(ext)
            library_dirs = copy.deepcopy(ext.library_dirs)
            if with_sdaa:
                libraries.append(f"teco_{ext.name}")
                library_dirs.append(os.path.abspath(self.build_temp))
            self.compiler.link_shared_object(
                c_objects,
                ext_path,  # output_filename
                libraries=libraries,
                library_dirs=library_dirs,
                runtime_library_dirs=library_dirs + ext.runtime_library_dirs,
                extra_postargs=CC_LINK_FLAGS,
                export_symbols=self.get_export_symbols(ext),
                debug=self.debug,
                build_temp=self.build_temp,
                target_lang=language,
            )

        unix_single_compile(sources)

    def build_extensions(self):
        self._check_abi()

        # Note(Aurelius84): If already compiling source before, we should check whether
        # cflags have changed and delete the built shared library to re-compile the source
        # even though source file content keep unchanged.
        so_name = self.get_ext_fullpath(self.extensions[0].name)
        clean_object_if_change_cflags(os.path.abspath(so_name), self.extensions[0])

        with_sdaa = any(is_sdaa_file(source) for source in self.extensions[0].sources)
        if with_sdaa:
            self.build_extension = self.sdaa_build_extension

        # Consider .scpp as valid source extensions.
        self.compiler.src_extensions += [".scpp"]
        # Save the original _compile method for later.
        original_compile = self.compiler._compile

        def unix_custom_single_compiler(
            obj, src, ext, cc_args, extra_postargs, pp_opts
        ):
            """
            Monkey patch mechanism to replace inner compiler to custom compile process on Unix platform.
            """
            # use abspath to ensure no warning and don't remove deepcopy because modify params
            # with dict type is dangerous.
            src = os.path.abspath(src)
            cflags = copy.deepcopy(extra_postargs)
            try:
                original_compiler = self.compiler.compiler_so
                if is_sdaa_file(src):
                    assert (
                        SDAA_HOME is not None
                    ), "Not found SDAA runtime, \
                            please use `export SDAA_HOME= XXX` to specify it."

                    # {'tecocc': {}, 'cxx: {}}
                    if isinstance(cflags, dict):
                        cflags = cflags["tecocc"]
                    cflags = prepare_unix_sdaaflags(cflags)
                # cxx compile Cpp source
                elif isinstance(cflags, dict):
                    cflags = cflags["cxx"]

                # NOTE(Aurelius84): Since Paddle 2.0, we require gcc version > 5.x,
                # so we add this flag to ensure the symbol names from user compiled
                # shared library have same ABI suffix with libpaddle.so.
                # See https://stackoverflow.com/questions/34571583/understanding-gcc-5s-glibcxx-use-cxx11-abi-or-the-new-abi
                add_compile_flag(cflags, ["-D_GLIBCXX_USE_CXX11_ABI=1"])

                add_std_without_repeat(
                    cflags, self.compiler.compiler_type, use_std14=True
                )
                original_compile(obj, src, ext, cc_args, cflags, pp_opts)
            finally:
                # restore original_compiler
                self.compiler.set_executable("compiler_so", original_compiler)

        def object_filenames_with_sdaa(origina_func, build_directory):
            """
            Decorated the function to add customized naming mechanism.
            Originally, both .cc/.cu will have .o object output that will
            bring file override problem. Use .cu.o as CUDA object suffix.
            """

            def wrapper(source_filenames, strip_dir=0, output_dir=""):
                try:
                    objects = origina_func(source_filenames, strip_dir, output_dir)
                    for i, source in enumerate(source_filenames):
                        # modify xx.o -> xx.scpp.o
                        if is_sdaa_file(source):
                            old_obj = objects[i]
                            objects[i] = old_obj[:-1] + "scpp.o"
                    # if user set build_directory, output objects there.
                    if build_directory is not None:
                        objects = [
                            os.path.join(build_directory, os.path.basename(obj))
                            for obj in objects
                        ]
                    # ensure to use abspath
                    objects = [os.path.abspath(obj) for obj in objects]
                finally:
                    self.compiler.object_filenames = origina_func

                return objects

            return wrapper

        # customized compile process
        self.compiler._compile = unix_custom_single_compiler

        self.compiler.object_filenames = object_filenames_with_sdaa(
            self.compiler.object_filenames, self.build_lib
        )
        self._record_op_info()

        print("Compiling user custom op, it will cost a few seconds.....")
        build_ext.build_extensions(self)

        # Reset runtime library path on MacOS platform
        so_path = self.get_ext_fullpath(self.extensions[0]._full_name)
        _reset_so_rpath(so_path)

    def get_ext_filename(self, fullname):
        # for example: customized_extension.cpython-37m-x86_64-linux-gnu.so
        ext_name = super().get_ext_filename(fullname)
        split_str = "."
        name_items = ext_name.split(split_str)
        if self.no_python_abi_suffix:
            assert (
                len(name_items) > 2
            ), f"Expected len(name_items) > 2, but received {len(name_items)}"
            name_items.pop(-2)
            ext_name = split_str.join(name_items)

        return ext_name

    def _check_abi(self):
        """
        Check ABI Compatibility.
        """
        if hasattr(self.compiler, "compiler_cxx"):
            compiler = self.compiler.compiler_cxx[0]
        else:
            compiler = os.environ.get("CXX", "c++")

        check_abi_compatibility(compiler)

    def _record_op_info(self):
        """
        Record custom op information.
        """
        # parse shared library abs path
        outputs = self.get_outputs()
        assert len(outputs) == 1
        # multi operators built into same one .so file
        so_path = os.path.abspath(outputs[0])
        so_name = os.path.basename(so_path)

        for i, extension in enumerate(self.extensions):
            sources = [os.path.abspath(s) for s in extension.sources]
            if not self.contain_sdaa_file:
                self.contain_sdaa_file = any(is_sdaa_file(s) for s in sources)
            op_names = parse_op_name_from(sources)

            for op_name in op_names:
                CustomOpInfo.instance().add(op_name, so_name=so_name, so_path=so_path)


class EasyInstallCommand(easy_install):
    """
    Extend easy_install Command to control the behavior of naming shared library
    file.

    NOTE(Aurelius84): This is a hook subclass inherited Command used to rename shared
                    library file after extracting egg-info into site-packages.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # NOTE(Aurelius84): Add args and kwargs to make compatible with PY2/PY3
    def run(self, *args, **kwargs):
        super().run(*args, **kwargs)
        # NOTE: To avoid failing import .so file instead of
        # python file because they have same name, we rename
        # .so shared library to another name.
        for egg_file in self.outputs:
            filename, ext = os.path.splitext(egg_file)
            will_rename = False
            if ext == ".so":
                will_rename = True

            if will_rename:
                new_so_path = filename + "_pd_" + ext
                if not os.path.exists(new_so_path):
                    os.rename(r"%s" % egg_file, r"%s" % new_so_path)
                assert os.path.exists(new_so_path)


class BuildCommand(build):
    """
    Extend build Command to control the behavior of specifying `build_base` root directory.

    NOTE(Aurelius84): This is a hook subclass inherited Command used to specify customized
                      build_base directory.
    """

    @classmethod
    def with_options(cls, **options):
        """
        Returns a BuildCommand subclass containing use-defined options.
        """

        class cls_with_options(cls):
            def __init__(self, *args, **kwargs):
                kwargs.update(options)
                cls.__init__(self, *args, **kwargs)

        return cls_with_options

    def __init__(self, *args, **kwargs):
        # Note: shall put before super()
        self._specified_build_base = kwargs.get("build_base", None)

        super().__init__(*args, **kwargs)

    def initialize_options(self):
        """
        build_base is root directory for all sub-command, such as
        build_lib, build_temp. See `distutils.command.build` for details.
        """
        super().initialize_options()
        if self._specified_build_base is not None:
            self.build_base = self._specified_build_base


def load(
    name,
    sources,
    extra_cxx_cflags=None,
    extra_sdaa_cflags=None,
    extra_ldflags=None,
    extra_include_paths=None,
    extra_library_paths=None,
    build_directory=None,
    verbose=False,
):
    """
    An Interface to automatically compile C++/CUDA source files Just-In-Time
    and return callable python function as other Paddle layers API. It will
    append user defined custom operators in background while building models.

    It will perform compiling, linking, Python API generation and module loading
    processes under a individual subprocess. It does not require CMake or Ninja
    environment. On Linux platform, it requires GCC compiler whose version is
    greater than 5.4 and it should be soft linked to ``/usr/bin/cc`` . On Windows
    platform, it requires Visual Studio whose version is greater than 2017.
    On MacOS, clang++ is requited. In addition, if compiling Operators supporting
    GPU device, please make sure ``nvcc`` compiler is installed in local environment.

    Moreover, `ABI compatibility <https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html>`_
    will be checked to ensure that compiler version from ``cc(Linux)`` , ``cl.exe(Windows)``
    on local machine is compatible with pre-installed Paddle whl in python site-packages.

    For Linux, GCC version will be checked . For example if Paddle with CUDA 10.1 is built with GCC 8.2,
    then the version of user's local machine should satisfy GCC >= 8.2.
    For Windows, Visual Studio version will be checked, and it should be greater than or equal to that of
    PaddlePaddle (Visual Studio 2017).
    If the above conditions are not met, the corresponding warning will be printed, and a fatal error may
    occur because of ABI compatibility.

    Compared with ``setup`` interface, it doesn't need extra ``setup.py`` and excute
    ``python setup.py install`` command. The interface contains all compiling and installing
    process underground.

    Note:

        1. Currently we support Linux, MacOS and Windows platform.
        2. On Linux platform, we recommend to use GCC 8.2 as soft linking candidate of ``/usr/bin/cc`` .
           Then, Use ``which cc`` to ensure location of ``cc`` and using ``cc --version`` to ensure linking
           GCC version.
        3. On Windows platform, we recommend to install `` Visual Studio`` (>=2017).


    **A simple example:**

    .. code-block:: text

        import paddle
        from paddle.utils.cpp_extension import load

        custom_op_module = load(
            name="op_shared_libary_name",                # name of shared library
            sources=['relu_op.cc', 'relu_op.cu'],        # source files of customized op
            extra_cxx_cflags=['-g', '-w'],               # optional, specify extra flags to compile .cc/.cpp file
            extra_cuda_cflags=['-O2'],                   # optional, specify extra flags to compile .cu file
            verbose=True                                 # optional, specify to output log information
        )

        x = paddle.randn([4, 10], dtype='float32')
        out = custom_op_module.relu(x)


    Args:
        name(str): Specify the name of generated shared library file name, not including ``.so`` and ``.dll`` suffix.
        sources(list[str]): Specify source files name of customized operators.  Supporting ``.cc`` , ``.cpp`` for CPP file
                            and ``.cu`` for CUDA file.
        extra_cxx_cflags(list[str], optional): Specify additional flags used to compile CPP files. By default
                               all basic and framework related flags have been included.
        extra_cuda_cflags(list[str], optional): Specify additional flags used to compile CUDA files. By default
                               all basic and framework related flags have been included.
                               See `Cuda Compiler Driver NVCC <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html>`_
                               for details. Default is None.
        extra_ldflags(list[str], optional): Specify additional flags used to link shared library. See
                                `GCC Link Options <https://gcc.gnu.org/onlinedocs/gcc/Link-Options.html>`_ for details.
                                Default is None.
        extra_include_paths(list[str], optional): Specify additional include path used to search header files. By default
                                all basic headers are included implicitly from ``site-package/paddle/include`` .
                                Default is None.
        extra_library_paths(list[str], optional): Specify additional library path used to search library files. By default
                                all basic libraries are included implicitly from ``site-packages/paddle/libs`` .
                                Default is None.
        build_directory(str, optional): Specify root directory path to put shared library file. If set None,
                            it will use ``PADDLE_EXTENSION_DIR`` from os.environ. Use
                            ``paddle.utils.cpp_extension.get_build_directory()`` to see the location. Default is None.
        verbose(bool, optional): whether to verbose compiled log information. Default is False.

    Returns:
        Module: A callable python module contains all CustomOp Layer APIs.

    """

    if build_directory is None:
        build_directory = get_build_directory(verbose)

    # ensure to use abs path
    build_directory = os.path.abspath(build_directory)

    log_v(f"build_directory: {build_directory}", verbose)

    file_path = os.path.join(build_directory, f"{name}_setup.py")
    sources = [os.path.abspath(source) for source in sources]

    if extra_cxx_cflags is None:
        extra_cxx_cflags = []
    if extra_sdaa_cflags is None:
        extra_sdaa_cflags = []
    assert isinstance(
        extra_cxx_cflags, list
    ), f"Required type(extra_cxx_cflags) == list[str], but received {extra_cxx_cflags}"
    assert isinstance(
        extra_sdaa_cflags, list
    ), "Required type(extra_sdaa_cflags) == list[str], but received {}".format(
        extra_sdaa_cflags
    )

    log_v(
        "additional extra_cxx_cflags: [{}], extra_sdaa_cflags: [{}]".format(
            " ".join(extra_cxx_cflags),
            " ".join(extra_sdaa_cflags),
        ),
        verbose,
    )

    # write setup.py file and compile it
    build_base_dir = os.path.join(build_directory, name)

    _write_setup_file(
        name,
        sources,
        file_path,
        build_base_dir,
        extra_include_paths,
        extra_library_paths,
        extra_cxx_cflags,
        extra_sdaa_cflags,
        extra_ldflags,
        verbose,
    )
    _jit_compile(file_path, verbose)

    # import as callable python api
    custom_op_api = _import_module_from_library(name, build_base_dir, verbose)

    return custom_op_api
