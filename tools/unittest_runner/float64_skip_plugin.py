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

"""
float64_skip_plugin.py

pytest plugin to skip test cases that use float64 data type.
When the environment variable FLAG_SKIP_FLOAT64 is set to 1, this plugin will skip
all test cases that use float64, unless the child class overrides the attribute
or method and removes float64 usage.
"""

import inspect
import os
import traceback

import numpy as np
import pytest


def _get_func_source(val):
    """Return the source of a callable or property method."""
    try:
        if isinstance(val, (staticmethod, classmethod)):
            return inspect.getsource(val.__func__)
        elif isinstance(val, property):
            srcs = []
            for f in (val.fget, val.fset, val.fdel):
                if f is not None:
                    try:
                        srcs.append(inspect.getsource(f))
                    except Exception:
                        pass
            return "\n".join(srcs)
        else:
            return inspect.getsource(val)
    except Exception:
        return ""


def _attr_contains_float64(val):
    """Check whether attribute value contains float64."""
    try:
        # Treat complex128 as a float64-related dtype as well
        if val is np.float64 or val is np.complex128:
            return True
        if isinstance(val, np.dtype) and (
            val == np.dtype("float64") or val == np.dtype("complex128")
        ):
            return True
        s = repr(val)
        return ("float64" in s) or ("complex128" in s)
    except Exception:
        return False


def _callable_contains_float64(val):
    """Check whether a callable or property method contains float64 in its source."""
    src = _get_func_source(val)
    # Also consider complex128 usage as part of float64-like tests
    return ("float64" in src) or ("complex128" in src)


def _child_overrides_without_float64(test_class, name):
    """
    If child class defines the same attribute/method name, and its version
    does NOT contain float64, return True (meaning: don't skip).
    """
    if name not in test_class.__dict__:
        return False  # child does NOT override

    child_val = test_class.__dict__[name]

    # callable or property?
    if callable(child_val) or isinstance(
        child_val, (staticmethod, classmethod, property)
    ):
        return not _callable_contains_float64(child_val)

    # normal attribute: check if float64-free
    return not _attr_contains_float64(child_val)


def pytest_configure(config):
    """
    Monkey-patch numpy.random.random to generate float32.
    """
    original_random = np.random.random

    def random_float32(size=None):
        return original_random(size).astype(np.float32)

    np.random.random = random_float32


def pytest_collection_modifyitems(config, items):
    """
    Skip tests whose class or base classes (up to but NOT including OpTest) contain
    'float64' in method source or class attributes, unless the child class overrides
    the attribute and removes float64 usage.
    """

    skip_float64 = os.environ.get("FLAG_SKIP_FLOAT64", "0") == "1"
    if not skip_float64:
        return

    for item in items:
        try:
            test_class = getattr(item, "cls", None)
            if test_class is None:
                continue

            skip_test = False
            debug_info = None

            # Walk class MRO (stop at OpTest or object)
            for cls in inspect.getmro(test_class):
                if cls is object or cls.__name__ == "OpTest":
                    break

                for name, val in cls.__dict__.items():
                    if name.startswith("_"):
                        continue

                    # ----- Case 1: callable / method / property -----
                    if callable(val) or isinstance(
                        val, (staticmethod, classmethod, property)
                    ):
                        if _callable_contains_float64(val):

                            # Check override in child class
                            if _child_overrides_without_float64(test_class, name):
                                continue  # child removes float64 → safe

                            skip_test = True
                            debug_info = (cls, name, "callable/property", val)
                            break

                    # ----- Case 2: normal attribute -----
                    else:
                        if _attr_contains_float64(val):

                            # Check override in child class
                            if _child_overrides_without_float64(test_class, name):
                                continue  # child removes float64 → safe

                            skip_test = True
                            debug_info = (cls, name, "attribute", val)
                            break

                if skip_test:
                    break

            if skip_test:
                try:
                    cls, name, kind, what = debug_info
                    print(
                        f"[SKIP-FLOAT64] Skipping test {item.nodeid}: detected 'float64/complex128' in {kind} "
                        f"'{name}' of class {cls.__module__}.{cls.__name__}. (repr: {what!r})"
                    )
                except Exception:
                    print(
                        f"[SKIP-FLOAT64] Skipping test {item.nodeid}: detected 'float64/complex128'"
                    )

                item.add_marker(pytest.mark.skip(reason="SKIP FLOAT64 TESTS"))

        except Exception:
            traceback.print_exc()
            continue
