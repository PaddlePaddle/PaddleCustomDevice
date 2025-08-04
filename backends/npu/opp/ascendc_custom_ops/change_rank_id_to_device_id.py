import json
import os
from functools import partial
from argparse import ArgumentParser


FILTER_DIRS = [".profiler", "HCCL_PROF", "timeline", "query", 'sqlite', 'log']


def get_path_dir(path: str) -> list:
    """
    check result path exist JOB dir
    path : result path
    """
    path_dir_filter = filter(partial(_path_dir_filter_func, root_dir=path), os.listdir(path))
    sub_dirs = list(path_dir_filter)
    if not sub_dirs:
        message = f"The path \"{path}\" does not have PROF dir. Please check the path."
        print(message)
    return sub_dirs


def _path_dir_filter_func(sub_path, root_dir):
    return sub_path not in FILTER_DIRS and os.path.isdir(os.path.realpath(os.path.join(root_dir, sub_path)))


def change_rank_id_to_device_id(pro_dir):
    for root, dirs, files in os.walk(pro_dir):
        for dir_ in dirs:
            if 'device_' in dir_:
                device_id = dir_.split("_")[-1]

                info_path = os.path.join(root, dir_, f'info.json.{device_id}')

                with open(info_path, 'r+') as f:
                    info_data = json.load(f)
                print("ori: ", info_data.get('rank_id'))
                info_data['rank_id'] = int(device_id)
                print("modify: ", info_data['rank_id'])
                with open(info_path, "w+") as f:
                    json.dump(info_data, f)


def set_rank_id(pro_dir, rank_id_set):
    for root, dirs, files in os.walk(pro_dir):
        for dir_ in dirs:
            if 'device_' in dir_:
                device_id = dir_.split("_")[-1]

                info_path = os.path.join(root, dir_, f'info.json.{device_id}')

                with open(info_path, 'r+') as f:
                    info_data = json.load(f)
                print("ori: ", info_data.get('rank_id'))
                idx = 0
                while int(device_id) + idx * 8 in rank_id_set:
                    idx += 1
                rank_id = int(device_id) + idx * 8
                rank_id_set.add(rank_id)
                info_data['rank_id'] = rank_id
                print("modify: ", info_data['rank_id'])
                with open(info_path, "w+") as f:
                    json.dump(info_data, f)


def get_node_id_set_rank_id(dir_path):
    for dir_name in os.listdir(dir_path):
        node_id = dir_name.split("_")[0]
        for root, dirs, files in os.walk(os.path.join(dir_path, dir_name)):
            for dir_ in dirs:
                if 'device_' in dir_:
                    device_id = dir_.split("_")[-1]

                    info_path = os.path.join(root, dir_, f'info.json.{device_id}')

                    with open(info_path, 'r+') as f:
                        info_data = json.load(f)
                    print("ori: rank_id:", info_data.get('rank_id'), "device_id: ", device_id, "node_id: ", node_id)
                    rank_id = int(device_id) + int(node_id) * 8
                    info_data['rank_id'] = rank_id
                    print("modify: ", info_data['rank_id'])
                    with open(info_path, "w+") as f:
                        json.dump(info_data, f)


def set_soft_link(args):

    cmd = "find %s -name PROF* | xargs  -I {} mv {} %s" % (args.data, args.output)
    print("rm_data cmd:{} begin".format(cmd))
    os.system(cmd)
    print("rm_data cmd:{} end".format(cmd))


def parse_args():
    parser = ArgumentParser(description="Merge timeline for multi card")
    parser.add_argument("--data", "-d", default=None, help="root dir of PROF_* data")
    parser.add_argument("--output", default=None, help="soft link dir")
    arg = parser.parse_args()
    return arg


if __name__ == "__main__":
    args = parse_args()
    print(" ======================== set rank id and soft link ========================")
    change_rank_id_to_device_id(args.data)
    # get_node_id_set_rank_id(args.data)
    # set_soft_link(args)
