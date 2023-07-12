import argparse
import os
import shutil
import subprocess
import sys

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
    '--accum_iter', 
    type=int, 
    default=None,
    help="Accumulative iterations"
)

parser.add_argument(
    '--base_lr', 
    type=float, 
    default=0.0001,
    help="Base learning rate"
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
    '--restart_optimizer', action='store_true', help='Restart the optimizer between frames.')
parser.set_defaults(restart_optimizer=False)

parser.add_argument(
    '--ckpt_iters',
    type=int,
    default=0
)

parser.add_argument(
    '--resume', action='store_true', help='whether to resume from checkpoint or not')
parser.set_defaults(resume=False)

args = parser.parse_args()


if __name__ == '__main__':
    # exp_log_dir = os.path.join('exp_dir', os.path.basename(args))
    # model_output_dir = os.path.join('output', os.path.basename(args.ttt_output))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    detectron2_root = os.getenv("DETECTRON2_DATASETS")
    assert detectron2_root is not None, 'Need to set $DETECTRON2_DATASETS enviroment variable!'

    for vid in args.videos:
        in_dir = os.path.join(detectron2_root, "coco_videos", "raw_images", vid)
        train_dir = os.path.join(detectron2_root, "coco_videos", "raw_images", vid)
        root_dir = os.path.join(args.output_dir, 
                                vid + "_" +
                                "instance_" +
                                str(args.base_lr) + "_" +
                                str(args.batch_size)
                            )
        os.makedirs(root_dir, exist_ok=True)


        # 1. Create output folder and copy relevant data over from dataset root
        data_dir = os.path.join(root_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        data_train_dir = os.path.join(data_dir, "train")
        if os.path.exists(data_train_dir):
            shutil.rmtree(data_train_dir)
        shutil.copytree(train_dir, data_train_dir)

        run_args = [
            sys.executable, "train_ttt_mae.py",
            "--num-gpus", "1",
            "--config-file", "configs/coco_videos/instance-segmentation/swin/maskformer2_swin_small_bs16_50ep_ttt_mae.yaml",
            "--drop_aug",
            "--ttt_in_dir", in_dir,
            "--ttt_out_dir", data_dir,
            "--exp_dir", "exp_dir/mae_coco_inst_" +
                            str(args.batch_size) + "_" + 
                            str(args.base_lr)
        ]
        resume_args = ["--resume"] if args.resume else []
        detectron2_args = [
            "MODEL.WEIGHTS", args.weights,
            "OUTPUT_DIR", root_dir,
            "SOLVER.IMS_PER_BATCH", str(args.batch_size),
            "SOLVER.ACCUM_ITER", str(args.accum_iter),
            "SOLVER.BASE_LR", str(args.base_lr),
            "TTT.USE_SEG_HEAD", "False",
            "TTT.RESTART_OPTIMIZER", 'True' if args.restart_optimizer else 'False',
            "TTT.COCO_VID", vid,
            "TTT.CHECKPOINT_ITERS", str(args.ckpt_iters),
        ]

        all_args = run_args + resume_args + detectron2_args
        subprocess.run(all_args, check=True)

        # Remove the appropriate directory after training
        if os.path.exists(root_dir):
            shutil.rmtree(root_dir)
            


