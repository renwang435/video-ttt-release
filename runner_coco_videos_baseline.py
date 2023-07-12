import shutil
import argparse
import subprocess
import os
import sys
import numpy as np

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument(
    '--gpu',
    type=str,
    default=None,
    help="CUDA_VISIBLE_DEVICES parameter"
)

parser.add_argument(
    '--videos', 
    nargs='+', 
    type=str, 
    default=None,
    help="Top k% most confident"
)


parser.add_argument(
    '--batch_size', 
    type=int, 
    default=None,
    help="Batch size"
)


parser.add_argument(
    '--weights',
    type=str,
    default="../../../../checkpoints/maskformer_swin_s_sem_cityscapes_maskpatch16_joint.pkl",
)

parser.add_argument(
    '--output_dir',
    type=str,
    default="output",
)

parser.add_argument(
    '--eval_type',
    type=str,
    choices=["inst", "pano"],
    default=None,
)

parser.add_argument(
    "--num_imgs",
    type=int,
    default=None
)


parser.set_defaults(restart_optimizer=False)

args = parser.parse_args()


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    detectron2_root = os.getenv("DETECTRON2_DATASETS")
    assert detectron2_root is not None, 'Need to set $DETECTRON2_DATASETS enviroment variable!'

    if args.eval_type == "inst":
        config_file = "configs/coco_videos/instance-segmentation/swin/maskformer2_swin_small_bs16_50ep_ttt_mae.yaml"
        dataset_test = "coco_videos_instance_baseline"
    else:
        config_file = "configs/coco_videos/panoptic-segmentation/swin/maskformer2_swin_small_bs16_50ep_ttt_mae.yaml"
        dataset_test = "coco_videos_panoptic_baseline"
    num_imgs = str(args.num_imgs) if args.num_imgs is not None else "all"

    for vid in args.videos:
        in_dir = os.path.join(detectron2_root, "coco_videos", "raw_images", vid)
        train_dir = os.path.join(detectron2_root, "coco_videos", "raw_images", vid)
        root_dir = os.path.join(args.output_dir,
                                num_imgs,
                                vid
                            )
        # NEED TO CLEAR CACHE IF IT EXISTS
        if os.path.exists(root_dir):
            shutil.rmtree(root_dir)
        os.makedirs(root_dir, exist_ok=True)


        # 1. Create output folder and copy relevant data over from dataset root
        data_dir = os.path.join(root_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        data_train_dir = os.path.join(data_dir, "train")
        if os.path.exists(data_train_dir):
            shutil.rmtree(data_train_dir)
        shutil.copytree(train_dir, data_train_dir)

        subprocess.run([
            sys.executable, "eval_only_coco_videos.py",
            "--num-gpus", "1",
            "--config-file", config_file,
            "--drop_aug",
            "--ttt_in_dir", in_dir,
            "--ttt_out_dir", data_dir,
            "MODEL.WEIGHTS", args.weights,
            "OUTPUT_DIR", root_dir,
            "SOLVER.IMS_PER_BATCH", str(args.batch_size),
            "TTT.USE_SEG_HEAD", "False",
            "TTT.NUM_BASELINE_IMGS", "None" if args.num_imgs is None else num_imgs,
            "TTT.COCO_VID", vid,
            "DATASETS.TEST", '(\"' + dataset_test + '\",)'
        ], check=True)

        # # Remove the appropriate directory after training
        # if os.path.exists(root_dir):
        #     shutil.rmtree(root_dir)
