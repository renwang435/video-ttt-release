import argparse
import json
import os
import sys

from detectron2.utils.file_io import PathManager
from detectron2.data import (DatasetCatalog, MetadataCatalog)

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
]


def get_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
            Examples:

            Run on single machine:
                $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml

            Change some config options:
                $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

            Run on multiple machines:
                (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
                (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
            """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="""
            Modify config options at the end of the command. For Yacs configs, use
            space-separated "PATH.KEY VALUE" pairs.
            For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )

    # TTT arguments
    parser.add_argument(
        "--ttt_in_dir",
        type=str,
        default="",
        help=""
    )

    parser.add_argument(
        "--ttt_out_dir",
        type=str,
        default="",
        help=""
    )

    parser.add_argument(
        '--exp_dir',
        type=str,
        default=None,
        help='Experiment directory to save logs'
    )

    parser.add_argument(
        "--ttt_topl",
        type=float,
        default=None,
        help="Top fraction of confident pixels"
    )

    parser.add_argument(
        "--ttt_setting",
        type=str,
        default=None,
        help="Online or standard"
        
    )

    parser.add_argument(
        "--tent_setting",
        type=str,
        default=None,
        help="tent, bn, class-bal"
        
    )

    parser.add_argument(
        '--cbt', 
        type=float, 
        default=None,
        help="class balance reset threshold"
    )

    parser.add_argument(
        "--st_iters",
        type=int,
        default=None,
        help="Number of iterations of self-training for each image"
    )

    parser.add_argument(
        "--win_size",
        type=str,
        default=None,
    )


    # Dropout augmentation arguments
    parser.add_argument(
        "--drop_aug",
        action="store_true"
    )

    parser.add_argument(
        "--drop_ratio",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--mask_type",
        type=str,
        default=None
    )

    return parser


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


def _get_kitti_step_files(image_dir, queue, it):
    files = []

    # import ipdb; ipdb.set_trace()

    for img_idx in queue:
        img_root = format(img_idx, "06d")
        image_file = os.path.join(image_dir, img_root + '.png')

        files.append((image_file))
    assert len(files), "No images found in {}".format(image_dir)
    assert PathManager.isfile(files[0]), files[0] + ' is not a file'
    return files


def load_kitti_step_semantic(image_dir, queue, it):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g., "~/cityscapes/gtFine/train".
    Returns:
        list[dict]: a list of dict, each has "file_name" and
            "sem_seg_file_name".
    """
    ret = []
    for image_file in _get_kitti_step_files(image_dir, queue, it):
        ret.append(
            {
                "file_name": image_file,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"

    return ret


def _get_kitti_step_val_files(image_dir, ref_dir, img_idx):
    img_root = format(img_idx, "06d")

    files = []
        
    label_file = os.path.join(ref_dir, img_root + "_sem.png")
    image_file = os.path.join(image_dir, img_root + '.png')
    json_file = os.path.join(ref_dir, img_root + '.json')

    files.append((image_file, label_file, json_file))
    assert len(files), "No images found in {}".format(image_dir)
    for f in files[0]:
        assert PathManager.isfile(f), f
    return files


def load_kitti_video_eval(image_dir, ref_dir, img_idx):
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
    ref_dir = PathManager.get_local_path(ref_dir)
    for image_file, label_file, json_file, in _get_kitti_step_val_files(image_dir, ref_dir, img_idx):
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


def register_custom_kitti(ref_dir, queue, it):
    meta = _get_kitti_step_meta()

    image_dir = os.path.join(ref_dir, "train")
    gt_dir = os.path.join(ref_dir, "val")
    
    # Register training set
    name = f"kitti_step_video_sem_seg_train"
    DatasetCatalog.register(
        name, lambda x=image_dir, it=it: load_kitti_step_semantic(x, queue, it)
    )
    MetadataCatalog.get(name).set(
        stuff_classes=meta["stuff_classes"][:],
        evaluator_type="sem_seg",
        ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
    )
    
    # Register val set
    name = f"kitti_step_video_sem_seg_val"
    DatasetCatalog.register(
        name, lambda x=image_dir, y=gt_dir: load_kitti_video_eval(x, y, queue[-1])
    )
    MetadataCatalog.get(name).set(
        stuff_classes=meta["stuff_classes"][:],
        evaluator_type="sem_seg",
        ignore_label=255,  # NOTE: gt is saved in 16-bit TIFF images
    )

