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

from ..utils.utils import *  # noqa
from .version import *  # noqa
from paddle_sdaa.sdaa_ext import *  # noqa


def paddle_version():
    version = custom_paddle_compilation_version()
    return tensor_to_string(version)


def paddle_commit_version():
    version = custom_paddle_commit_compilation_version()
    return tensor_to_string(version)


def sdaa_runtime_version():
    version = custom_sdaa_runtime_version()
    return tensor_to_string(version)


def sdaa_driver_version():
    version = custom_sdaa_driver_version()
    return tensor_to_string(version)


def teco_dnn_version():
    version = custom_teco_dnn_version()
    return tensor_to_string(version)


def teco_blas_version():
    version = custom_teco_blas_version()
    return tensor_to_string(version)


def teco_custom_version():
    version = custom_teco_custom_version()
    return tensor_to_string(version)


def teco_tccl_version():
    version = custom_tccl_version()
    return tensor_to_string(version)


def sdpti_version():
    try:
        version = custom_sdpti_version()
    except:
        version = []

    return tensor_to_string(version)


def show():
    """Get the version of paddle-sdaa's dependencies and corresponding commit id.

    Returns:
        the following information will be output.

        tecodnn_version: the version of tecodnn

        tecoblas_version: the version of tecoblas

        sdaart_version: the version of sdaart

        sdaadriver_version: the version of sdaadriver

        commit_id: the commit_id of paddle-sdaa

    Examples:
        .. code-block:: python

            import paddle_sdaa

            paddle_sdaa.version.show()
            # tecodnn: 1.15.0
            # tecoblas: 1.15.0
            # tecocustom: 1.15.0
            # tccl: 1.14.0
            # sdaart: 1.0.0
            # sdaadriver: 1.0.0
            # commit: 078a943
    """
    print("tecodnn:", teco_dnn)
    print("tecoblas:", teco_blas)
    print("tecocustom:", teco_custom)
    print("tccl:", teco_tccl)
    print("sdaart:", sdaa_runtime)
    print("sdaadriver:", sdaa_driver)
    print("commit:", paddle_sdaa_commit)


def tecodnn():
    """Get tecodnn version of paddle-sdaa package.

    Returns:
        string: Return the version information of tecodnn.

    Examples:
        .. code-block:: python

            import paddle_sdaa

            paddle_sdaa.version.tecodnn()
            # '1.15.0'

    """
    return teco_dnn


def tecoblas():
    """Get tecoblas version of paddle-sdaa package.

    Returns:
        string: Return the version information of tecoblas.

    Examples:
        .. code-block:: python

            import paddle_sdaa

            paddle_sdaa.version.tecoblas()
            # '1.15.0'

    """
    return teco_blas


def tecocustom():
    """Get teco custom version of paddle-sdaa package.

    Returns:
        string: Return the version information of teco custom.

    Examples:
        .. code-block:: python

            import paddle_sdaa

            paddle_sdaa.version.tecocustom()
            # '1.15.0'

    """
    return teco_custom


def tccl():
    """Get tccl version of paddle-sdaa package.

    Returns:
        string: Return the version information of tccl.

    Examples:
        .. code-block:: python

            import paddle_sdaa

            paddle_sdaa.version.tccl()
            # '1.14.0'

    """
    return teco_tccl


def sdaart():
    """Get sdaart version of paddle-sdaa package.

    Returns:
        string: Return the version information of sdaart.

    Examples:
        .. code-block:: python

            import paddle_sdaa

            paddle_sdaa.version.sdaart()
            # '1.0.0'

    """
    return sdaa_runtime


def sdaadriver():
    """Get sdaadriver version of paddle-sdaa package.

    Returns:
        string: Return the version information of sdaadriver.

    Examples:
        .. code-block:: python

            import paddle_sdaa

            paddle_sdaa.version.sdaadriver()
            # '1.0.0'

    """
    return sdaa_driver


def commit():
    """Get commit_id of paddle-sdaa package.

    Returns:
        string: Return the commit_id of paddle-sdaa.

    Examples:
        .. code-block:: python

            import paddle_sdaa

            paddle_sdaa.version.commit()
            # '078a9436964431a0b91291a31fd2963515392731'

    """
    return paddle_sdaa_commit
