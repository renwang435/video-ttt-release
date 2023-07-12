import logging
import os
from collections import defaultdict

import pycocotools.mask as mask_util
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
from pycocotools.coco import COCO

logger = logging.getLogger(__name__)

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


def _get_coco_videos_files(all_raw_imgs, queue):
    files = []
    for img_idx in queue:
        image_file = all_raw_imgs[img_idx]

        files.append(image_file)
    assert len(files), "No images found in all_raw_imgs: {}".format(all_raw_imgs)
    for f in files:
        assert PathManager.isfile(f), (f, all_raw_imgs, queue)
    return files


def load_coco_videos_instance(all_raw_imgs, queue):
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


def load_coco_val_json(coco_api, all_raw_imgs, img_idx):
    id_map = None
    meta = MetadataCatalog.get("coco_videos_instance_val")
    cat_ids = sorted(coco_api.getCatIds())
    cats = coco_api.loadCats(cat_ids)
    # The categories in a custom json file may not be sorted.
    thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
    meta.thing_classes = thing_classes

    id_map = {v: i for i, v in enumerate(cat_ids)}
    meta.thing_dataset_id_to_contiguous_id = id_map

    ##########################
    img_root = all_raw_imgs[img_idx].split('/')[-1]
    image_dir = '/'.join(all_raw_imgs[img_idx].split('/')[:-1])
    ##########################


    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    # Take on our relevant image ids
    img_ids = [i for i in img_ids if coco_api.imgs[i]['file_name'] == img_root]
    if len(img_ids) == 0:
        return []
    imgs = coco_api.loadImgs(img_ids)
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    assert len(anns[0]) > 0
    
    imgs_anns = list(zip(imgs, anns))

    dataset_dicts = []

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"]
    assert len(anns) == 1
    img_dict, anno_dict_list = imgs[0], anns[0]

    num_instances_without_valid_segmentation = 0
    record = {}
    record["file_name"] = os.path.join(image_dir, img_dict["file_name"])
    record["height"] = img_dict["height"]
    record["width"] = img_dict["width"]
    image_id = record["image_id"] = img_dict["id"]

    objs = []
    assert len(anno_dict_list) == 1, anno_dict_list
    assert anno_dict_list[0]["image_id"] == image_id
    for anno in anno_dict_list[0]['segments_info']:
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


def register_custom_coco_videos(coco_api, all_raw_imgs, queue, it):
    assert len(queue) > 0
    # Register training set
    name = f"coco_videos_instance_train"
    DatasetCatalog.register(
        name, lambda x=all_raw_imgs: load_coco_videos_instance(x, queue)
    )
    MetadataCatalog.get(name).set(
        ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
    )
    
    # Register val set
    name = f"coco_videos_instance_val"
    DatasetCatalog.register(
        name, lambda x=all_raw_imgs: load_coco_val_json(coco_api, x, queue[-1])
    )
    cat_ids = sorted(coco_api.getCatIds())
    cats = coco_api.loadCats(cat_ids)
    thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
    thing_colors = []
    for thing in thing_classes:
        for k in COCO_CATEGORIES:
            if thing == k['name']:
                thing_colors.append(k['color'])
    MetadataCatalog.get(name).set(
        evaluator_type='coco_videos_instance_seg',
        ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
        thing_classes=thing_classes,
        thing_colors=thing_colors,

    )


