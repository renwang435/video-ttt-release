import contextlib
import io
import logging
import os
from collections import defaultdict

import numpy as np
import pycocotools.mask as mask_util
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
from fvcore.common.timer import Timer
from pycocotools.coco import COCO

logger = logging.getLogger(__name__)

POSSIBLE_VAL_IMGS = np.arange(1, 4000, 10)


class COCOVideos(COCO):
    def __init__(self, annotation_file=None):
        super().__init__(annotation_file=annotation_file)
    
    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns,catToImgs = defaultdict(list),defaultdict(list)
        if 'annotations' in self.dataset:
            for i, ann in enumerate(self.dataset['annotations']):
                imgToAnns[ann['image_id']].append(ann)
                # anns[ann['id']] = ann
                anns[i] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img
        self.dataset['categories'] = COCO_CATEGORIES
        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                for seg_info in ann['segments_info']:
                    catToImgs[seg_info['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats


def load_coco_json(json_file, image_root, dataset_name, vid, max_img_idx):
    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCOVideos(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map
        
    max_files = [f + 1 for f in np.arange(max_img_idx) if (f + 1) in POSSIBLE_VAL_IMGS]
    img_ids = [k for k, v in coco_api.imgs.items() if int(v["file_name"].split('_')[-1].split('.png')[0]) in max_files]


    imgs = coco_api.loadImgs(img_ids)
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]


    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"]

    num_instances_without_valid_segmentation = 0

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        assert len(anno_dict_list) == 1, anno_dict_list
        assert anno_dict_list[0]["image_id"] == image_id
        for anno in anno_dict_list[0]["segments_info"]:
            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}
            if "bbox" in obj and len(obj["bbox"]) == 0:
                raise ValueError(
                    f"One annotation of image {image_id} contains empty 'bbox' value! "
                    "This json does not have valid COCO format."
                )

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj["segmentation"] = segm


            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if obj["category_id"] != 255:
                if id_map:
                    annotation_category_id = obj["category_id"]
                    try:
                        obj["category_id"] = id_map[annotation_category_id]
                    except KeyError as e:
                        raise KeyError(
                            f"Encountered category_id={annotation_category_id} "
                            "but this id does not exist in 'categories' of the json file."
                        ) from e
                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)


    return dataset_dicts


def register_coco_instances(name, json_file, image_root, vid, max_img_idx):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).
    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.
    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name, vid, max_img_idx))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        evaluator_type="coco", 
        ignore_label=255,
    )


def register_coco_videos_instance_baseline(root, vid, max_img_idx):
    root = os.path.join(root, "coco_videos")
    max_img_idx = 4000 if max_img_idx is None else max_img_idx

    name = "coco_videos_instance_baseline"
    image_dir = os.path.join(root, 'raw_images', vid)
    json_file = os.path.join(root, 'annotations', 'train', vid + '.json')
    register_coco_instances(name, json_file, image_dir, vid, max_img_idx)

