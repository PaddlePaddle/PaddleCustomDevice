# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import os  # Added to allow dynamic device_id configuration

import paddle
from paddle.base import core

device_id = int(os.getenv("HPU_DEVICE_ID", 0))


def memory_allocated(device_id: int | None = None) -> int:
    """
    Return the current size of hpu memory that is allocated to tensor of the given device.
    """
    name = "hpu.memory_allocated"

    if not paddle.is_compiled_with_custom_device("intel_hpu"):
        raise ValueError(f"The API {name} is not supported in CPU-only PaddlePaddle.")
    return core.device_memory_stat_current_value("Allocated", device_id)


def memory_reserved(device_id: int | None = None) -> int:
    """
    Return the current size of HPU memory that is held by the allocator of the given device.
    """
    name = "hpu.memory_reserved"
    if not paddle.is_compiled_with_custom_device("intel_hpu"):
        raise ValueError(f"The API {name} is not supported in CPU-only PaddlePaddle.")
    return core.device_memory_stat_current_value("Reserved", device_id)


def max_memory_allocated(device_id: int | None = None) -> int:
    """
    Return the peak size of HPU memory that was allocated to tensors on the given device.
    """
    name = "hpu.max_memory_allocated"
    if not paddle.is_compiled_with_custom_device("intel_hpu"):
        raise ValueError(f"The API {name} is not supported in CPU-only PaddlePaddle.")
    if device_id is None:
        raise ValueError(f"device_id must be specified for {name}.")
    return core.device_memory_stat_peak_value("Allocated", device_id)


def max_memory_reserved(device_id: int | None = None) -> int:
    """
    Return the peak size of HPU memory that is held by the allocator of the given device.
    """
    name = "max_memory_reserved"
    if not paddle.is_compiled_with_custom_device("intel_hpu"):
        raise ValueError(f"The API {name} is not supported in CPU-only PaddlePaddle.")
    return core.device_memory_stat_peak_value("Reserved", device_id)


class TestMemoryAllocated(unittest.TestCase):
    def test_memory_allocated(self):
        if paddle.is_compiled_with_custom_device("intel_hpu"):
            tensor = paddle.zeros(shape=[256])
            alloc_size = 2 * 256
            memory_allocated_size = memory_allocated(device_id)
            self.assertEqual(memory_allocated_size, alloc_size)
            print(f"TestMemoryAllocated: `Passed` for device_id={device_id}")
            del tensor  # Cleanup


class TestMemoryReserved(unittest.TestCase):
    def test_memory_reserved(self):
        if paddle.is_compiled_with_custom_device("intel_hpu"):
            tensor = paddle.zeros(shape=[256])
            alloc_size = 2 * 256  # 256 bfloat16 data, with 2 bytes for each one
            memory_reserved_size = memory_reserved(device_id)
            self.assertGreaterEqual(
                memory_reserved_size,
                alloc_size,
                f"Reserved memory ({memory_reserved_size}) is less than expected ({alloc_size}).",
            )
            print(f"TestMemoryReserved: `Passed` for device_id={device_id}")
            del tensor  # Cleanup


class TestMaxMemoryAllocated(unittest.TestCase):
    def test_max_memory_allocated(self):
        if paddle.is_compiled_with_custom_device("intel_hpu"):
            alloc_time = 100
            max_alloc_size = 10000
            peak_memory_allocated_size = max_memory_allocated(device_id)
            for _ in range(alloc_time):
                shape = [paddle.randint(1, max_alloc_size).numpy().item()]
                tensor = paddle.zeros(shape)
                peak_memory_allocated_size = max(
                    peak_memory_allocated_size, memory_allocated(device_id)
                )
                del tensor  # Cleanup

            self.assertEqual(
                peak_memory_allocated_size, max_memory_allocated(device_id)
            )
            print(
                f"TestMaxMemoryAllocated `Passed` peak memory allocated size: {peak_memory_allocated_size} for device_id={device_id}"
            )


class TestMaxMemoryReserved(unittest.TestCase):
    def test_max_memory_reserved(self):
        if paddle.is_compiled_with_custom_device("intel_hpu"):
            alloc_time = 100
            max_alloc_size = 10000
            peak_memory_reserved_size = max_memory_reserved(device_id)
            for _ in range(alloc_time):
                shape = [paddle.randint(1, max_alloc_size).numpy().item()]
                tensor = paddle.zeros(shape)
                peak_memory_reserved_size = max(
                    peak_memory_reserved_size, memory_reserved(device_id)
                )
                del tensor  # Cleanup

            self.assertEqual(peak_memory_reserved_size, max_memory_reserved(device_id))
            print(
                f"TestMaxMemoryReserved `Passed` peak meory reserved size: {peak_memory_reserved_size} for device_id={device_id}"
            )


if __name__ == "__main__":
    paddle.device.set_device(f"intel_hpu:{device_id}")
    paddle.set_default_dtype("bfloat16")

    unittest.main()
