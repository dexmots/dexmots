import os
import sapien
import json

token = os.environ.get("SAPIEN_TOKEN")


def download_part(part_id: int, part_name: str):
    urdf_file = sapien.asset.download_partnet_mobility(part_id, token)
    downloaded_dir, filename = os.path.split(urdf_file)
    results_file = os.path.join(downloaded_dir, "result.json")
    if part_name == "":
        part_name = get_object_name(results_file)
    if not os.path.exists(os.path.join("assets", part_name)):
        os.makedirs(os.path.join("assets", part_name))
    new_part_path = os.path.join("assets", part_name)
    os.rename(downloaded_dir, new_part_path)
    print("saved to {}".format(new_part_path))
    return new_part_path


def get_object_name(results_file_path: str):
    with open(results_file_path) as f:
        data = json.load(f)
    for elem in data:
        obj_class = elem["name"]

    obj_path = os.path.join(obj_class, os.path.split(os.path.dirname(results_file_path))[1])
    return obj_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--part_id", type=int, default=101463)  # 101463 -> spray_bottle
    parser.add_argument("--part_name", type=str, default="")  # 101463 -> spray_bottle
    args = parser.parse_args()
    download_part(args.part_id, args.part_name)
