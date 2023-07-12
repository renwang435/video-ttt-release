import json
import logging
import os
from copy import deepcopy

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.utils.file_io import PathManager

logger = logging.getLogger(__name__)
    

def get_metadata():
    meta = {}

    thing_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in COCO_CATEGORIES if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in COCO_CATEGORIES]
    stuff_colors = [k["color"] for k in COCO_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(COCO_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta

meta = get_metadata()

def _convert_category_id(segment_info):
    new_seg_info = deepcopy(segment_info)
    if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
        new_seg_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
            segment_info["category_id"]
        ]
        new_seg_info["isthing"] = True
    else:
        new_seg_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
            segment_info["category_id"]
        ]
        new_seg_info["isthing"] = False

    return new_seg_info


def _get_coco_videos_files(all_raw_imgs, queue):
    files = []
    for img_idx in queue:
        image_file = all_raw_imgs[img_idx]

        files.append(image_file)
    assert len(files), "No images found in all_raw_imgs: {}".format(all_raw_imgs)
    for f in files:
        assert PathManager.isfile(f), (f, all_raw_imgs, queue)
    return files


def load_coco_videos_panoptic(all_raw_imgs, queue):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g., "~/cityscapes/gtFine/train".
    Returns:
        list[dict]: a list of dict, each has "file_name" and
            "sem_seg_file_name".
    """
    ret = []
    files = _get_coco_videos_files(all_raw_imgs, queue)
    assert len(files) > 0
    for image_file in files:
        ret.append(
            {
                "file_name": image_file
            }
        )
    assert len(ret), f"No images found in all_raw_imgs: {all_raw_imgs}!"

    return ret


def load_coco_val_json_panoptic(all_raw_imgs, panoptic_json, img_idx):
    ##########################
    img_root = all_raw_imgs[img_idx].split('/')[-1]
    image_dir = '/'.join(all_raw_imgs[img_idx].split('/')[:-1])
    pan_label_dir = os.path.join('/'.join(image_dir.split('/')[:-2]), 'panoptic_images', image_dir.split('/')[-1])
    sem_seg_label_dir = os.path.join('/'.join(image_dir.split('/')[:-2]), 'semantic_images', image_dir.split('/')[-1])
    ##########################
    ret = []

    # Filter out the annotations we actually want for validation
    anns = [ann for ann in panoptic_json["annotations"] if ann["file_name"] == img_root]
    image_meta = [ann for ann in panoptic_json["images"] if ann["file_name"] == img_root]
    assert len(anns) == 1, 'Need exactly 1 annotation for TTT panoptic validation'
    assert len(image_meta) == 1

    image_id = int(anns[0]["image_id"])
    image_file = os.path.join(image_dir, img_root)
    pan_label_file = os.path.join(pan_label_dir, img_root)
    sem_seg_label_file = os.path.join(sem_seg_label_dir, img_root)
    segments_info = [_convert_category_id(x) for x in anns[0]["segments_info"] if x["category_id"] != 255]
    ret.append(
        {
            "file_name": image_file,
            "image_id": image_id,
            "pan_seg_file_name": pan_label_file,
            'sem_seg_file_name': sem_seg_label_file,
            "segments_info": segments_info,
            "height": image_meta[0]["height"],
            "width": image_meta[0]["width"],
        }
    )
    assert len(ret), f"No images found in all_raw_imgs: {all_raw_imgs}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
    assert PathManager.isfile(ret[0]["sem_seg_file_name"]), ret[0]["sem_seg_file_name"]
    return ret



def register_custom_coco_videos_panoptic(all_raw_imgs, queue, it):
    assert len(queue) > 0
    # Register training set
    name = f"coco_videos_panoptic_train"
    DatasetCatalog.register(
        name, lambda x=all_raw_imgs: load_coco_videos_panoptic(x, queue)
    )
    MetadataCatalog.get(name).set(
        ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
        **meta
    )

    img_path_split = all_raw_imgs[queue[-1]].split('/')
    img_root = img_path_split[-1]
    image_root = '/'.join(img_path_split[:-1])
    panoptic_root = os.path.join('/'.join(img_path_split[:-3]), 'panoptic_images', img_path_split[-2])
    sem_seg_root = os.path.join('/'.join(img_path_split[:-3]), 'semantic_images', img_path_split[-2])
    panoptic_json = os.path.join('/'.join(img_path_split[:-3]), 'annotations', 'train', img_path_split[-2] + '.json')
    instances_json = os.path.join('/'.join(img_path_split[:-3]), 'annotations', 'val', img_path_split[-2] + '.json')

    with open(panoptic_json, 'r') as fp:
        json_info = json.load(fp)
    anns = [ann for ann in json_info["annotations"] if ann["file_name"] == img_root]

    # Register val set
    name = f"coco_videos_panoptic_val_with_sem_seg"
    DatasetCatalog.register(
        name, lambda x=all_raw_imgs, y=json_info: load_coco_val_json_panoptic(x, y, queue[-1])
    )
    MetadataCatalog.get(name).set(
        sem_seg_root=sem_seg_root,
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        json_file=instances_json,
        evaluator_type="coco_videos_panoptic_seg",
        ignore_label=255,
        label_divisor=1000,
        gt_json={"annotations": anns, "categories" : COCO_CATEGORIES},
        **meta,
    )


