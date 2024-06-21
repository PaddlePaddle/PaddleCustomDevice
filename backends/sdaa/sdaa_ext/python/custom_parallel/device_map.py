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
from paddle_sdaa.sdaa_ext import *  # noqa
import paddle
import os


def device_core_map():
    """
    This function will get the the aicard nums of every core id.

    For example, it will get {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]} for two four-core aicards,
    it will get {0: [0, 1, 2], 1: [3, 4, 5]} for two three-core aicards.

    Returns:

        dict: the aicard nums of every core id
    """

    dummy_input = paddle.to_tensor([1], place=paddle.CPUPlace(), dtype="int32")
    device_card = rank_ids(dummy_input).numpy().tolist()
    card_core_map = dict()
    for idx, val in enumerate(device_card):
        cur_id_list = card_core_map.get(val, [])
        cur_id_list.append(idx)
        card_core_map[val] = cur_id_list
    return card_core_map


def get_cur_process_device_list():
    device_list = None
    if os.environ.get("SDAA_VISIBLE_DEVICES"):
        device_list = os.environ.get("SDAA_VISIBLE_DEVICES").split(",")
    else:
        device_list = paddle.device.get_available_device()
        device_list = [dev.split(":")[-1] for dev in device_list]
    mapped_device_id = os.environ.get("FLAGS_selected_sdaas")
    cur_device_id = int(device_list[int(mapped_device_id)])
    # get all physical device ids
    all_device_ids = []
    data = paddle.to_tensor([cur_device_id])
    task = paddle.distributed.stream.all_gather(all_device_ids, data, sync_op=False)
    task.wait()
    all_device_ids = [int(i) for i in all_device_ids]  # --devices, physical ids
    device_core_dict = device_core_map()
    devices_list = []
    for key, val in device_core_dict.items():
        cur_list = list(set(all_device_ids) & set(val))  # one physical card ids
        if cur_list != []:
            devices_list.append((all_device_ids, cur_list, cur_device_id))
    return devices_list
