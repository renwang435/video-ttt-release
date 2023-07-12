import json
import logging
import os
from copy import deepcopy

import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.utils.file_io import PathManager

logger = logging.getLogger(__name__)


POSSIBLE_VAL_IMGS = np.arange(1, 4000, 10)


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

    # Convert category id for training:
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


def load_coco_val_json_panoptic(image_dir, pan_label_dir, sem_seg_label_dir, anns, image_metas):
    ret = []

    for ann, image_meta in zip(anns, image_metas):
        # import ipdb; ipdb.set_trace()

        image_id = int(ann["image_id"])
        img_root = ann['file_name']
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
                "height": image_meta["height"],
                "width": image_meta["width"],
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
    assert PathManager.isfile(ret[0]["sem_seg_file_name"]), ret[0]["sem_seg_file_name"]
    return ret



def register_coco_videos_panoptic_baseline(root, vid, max_img_idx):
    image_root = os.path.join(root, 'coco_videos', 'raw_images', vid)
    panoptic_root = os.path.join(root, 'coco_videos', 'panoptic_images', vid)
    sem_seg_root = os.path.join(root, 'coco_videos', 'semantic_images', vid)
    panoptic_json = os.path.join(root, 'coco_videos', 'annotations', 'train', vid + '.json')
    instances_json = os.path.join(root, 'coco_videos', 'annotations', 'val', vid + '.json')

    with open(panoptic_json, 'r') as fp:
        json_info = json.load(fp)
    
    max_img_idx = 4000 if max_img_idx is None else max_img_idx
    max_files = [f + 1 for f in np.arange(max_img_idx) if (f + 1) in POSSIBLE_VAL_IMGS]
    anns = [ann for ann in json_info["annotations"] if int(ann["file_name"].split('_')[-1].split('.png')[0]) in max_files]
    image_meta = [ann for ann in json_info["images"] if int(ann["file_name"].split('_')[-1].split('.png')[0]) in max_files]

    # Register val set
    name = f"coco_videos_panoptic_baseline"
    DatasetCatalog.register(
        name, lambda x=image_root, y=panoptic_root, z=sem_seg_root, a=anns, b=image_meta: load_coco_val_json_panoptic(x, y, z, a, b)
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


