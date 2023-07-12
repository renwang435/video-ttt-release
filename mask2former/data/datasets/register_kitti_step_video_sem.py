# Copyright (c) Facebook, Inc. and its affiliates.
import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

KITTI_STEP_SEM_SEG_CATEGORIES = [
    {"name": "road", "id": 0, "trainId": 0},
    {"name": "sidewalk", "id": 1, "trainId": 1},
    {"name": "building", "id": 2, "trainId": 2},
    {"name": "wall", "id": 3, "trainId": 3},
    {"name": "fence", "id": 4, "trainId": 4},
    {"name": "pole", "id": 5, "trainId": 5},
    {"name": "traffic light", "id": 6, "trainId": 6},
    {"name": "traffic sign", "id": 7, "trainId": 7},
    {"name": "vegetation", "id": 8, "trainId": 8},
    {"name": "terrain", "id": 9, "trainId": 9},
    {"name": "sky", "id": 10, "trainId": 10},
    {"name": "person", "id": 11, "trainId": 11},
    {"name": "rider", "id": 12, "trainId": 12},
    {"name": "car", "id": 13, "trainId": 13},
    {"name": "truck", "id": 14, "trainId": 14},
    {"name": "bus", "id": 15, "trainId": 15},
    {"name": "train", "id": 16, "trainId": 16},
    {"name": "motorcycle", "id": 17, "trainId": 17},
    {"name": "bicycle", "id": 18, "trainId": 18},
    # {"name" : "unconfident", "id": 42, "trainId" : 42},     # ONLY FOR VISUALIZATION OF CONFIDENCE
]


def _get_kitti_step_meta():
    stuff_ids = [k["id"] for k in KITTI_STEP_SEM_SEG_CATEGORIES]

    # For semantic segmentation, this mapping maps from contiguous stuff id
    # (in [0, 91], used in models) to ids in the dataset (used for processing results)
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k["name"] for k in KITTI_STEP_SEM_SEG_CATEGORIES]

    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
    }
    return ret



def _get_kitti_step_files(image_dir, gt_dir, img_idx, it):
    img_root = format(img_idx, "06d")

    files = []
    # scan through the directory
    cities = PathManager.ls(gt_dir)
    for city in cities:
        city_img_dir = os.path.join(image_dir, city)
        city_gt_dir = os.path.join(gt_dir, city)

        label_file = os.path.join(city_gt_dir, img_root + "_" + str(it) + ".png")

        image_file = os.path.join(city_img_dir, img_root + '.png')
        json_file = os.path.join(city_gt_dir, img_root + '.json')

        files.append((image_file, label_file, json_file))
    assert len(files), "No images found in {}".format(image_dir)
    for f in files[0]:
        assert PathManager.isfile(f), f
    return files


def load_kitti_step_semantic(image_dir, gt_dir, img_idx, it):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g., "~/cityscapes/gtFine/train".
    Returns:
        list[dict]: a list of dict, each has "file_name" and
            "sem_seg_file_name".
    """
    ret = []
    # gt_dir is small and contain many small files. make sense to fetch to local first
    gt_dir = PathManager.get_local_path(gt_dir)
    for image_file, label_file, json_file in _get_kitti_step_files(image_dir, gt_dir, img_idx, it):
        with PathManager.open(json_file, "r") as f:
            jsonobj = json.load(f)
        ret.append(
            {
                "file_name": image_file,
                "sem_seg_file_name": label_file,
                "height": jsonobj["imgHeight"],
                "width": jsonobj["imgWidth"],
            }
        )
    assert len(ret), f"No images found in {image_dir}!"

    return ret



def _get_kitti_step_val_files(image_dir, gt_dir, img_idx):
    img_root = format(img_idx, "06d")

    files = []
    # scan through the directory
    cities = PathManager.ls(gt_dir)
    for city in cities:
        city_img_dir = os.path.join(image_dir, city)
        city_gt_dir = os.path.join(gt_dir, city)
        
        label_file = os.path.join(city_gt_dir, img_root + "_sem.png")
        image_file = os.path.join(city_img_dir, img_root + '.png')
        json_file = os.path.join(city_gt_dir, img_root + '.json')

        files.append((image_file, label_file, json_file))
    assert len(files), "No images found in {}".format(image_dir)
    for f in files[0]:
        assert PathManager.isfile(f), f
    return files


def load_kitti_video_eval(image_dir, gt_dir, img_idx):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g., "~/cityscapes/gtFine/train".
    Returns:
        list[dict]: a list of dict, each has "file_name" and
            "sem_seg_file_name".
    """
    ret = []
    # gt_dir is small and contain many small files. make sense to fetch to local first
    gt_dir = PathManager.get_local_path(gt_dir)
    for image_file, label_file, json_file in _get_kitti_step_val_files(image_dir, gt_dir, img_idx):
        with PathManager.open(json_file, "r") as f:
            jsonobj = json.load(f)
        ret.append(
            {
                "file_name": image_file,
                "sem_seg_file_name": label_file,
                "height": jsonobj["imgHeight"],
                "width": jsonobj["imgWidth"],
            }
        )
    assert len(ret), f"No images found in {image_dir}!"

    return ret    



def register_kitti_step_video_sem_seg(root, img_idx, it):
    root = os.path.join(root, "kitti_step")
    meta = _get_kitti_step_meta()

    for name, dirname in [("train", "train"),]:
        image_dir = os.path.join(root, "images", dirname)
        gt_dir = os.path.join(root, "st_video", dirname)
        name = f"kitti_step_video_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir, it=it: load_kitti_step_semantic(x, y, img_idx, it)
        )
        MetadataCatalog.get(name).set(
            stuff_classes=meta["stuff_classes"][:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
        )
    
    for name, dirname in [("val", "val")]:
        image_dir = os.path.join(root, "images", "train")
        gt_dir = os.path.join(root, "st_video", dirname)
        name = f"kitti_step_video_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir, it=it: load_kitti_video_eval(x, y, img_idx)
        )
        MetadataCatalog.get(name).set(
            stuff_classes=meta["stuff_classes"][:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
        )


_root = os.getenv("DETECTRON2_DATASETS")
register_kitti_step_video_sem_seg(_root, 0, 0)
