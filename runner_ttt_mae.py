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
    '--base_lrs', 
    nargs='+',
    type=float,
    default=[0.00001],
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
    '--optim',
    type=str,
    default='ADAMW'
)

parser.add_argument(
    '--update_every_n',
    type=str,
    default='1',
)


parser.add_argument(
    '--resume',
    action='store_true',
)
parser.set_defaults(resume=False)

parser.add_argument(
    '--restart_optimizer', action='store_true', help='Restart the optimizer between frames.')
parser.set_defaults(restart_optimizer=False)

args = parser.parse_args()


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    detectron2_root = os.getenv("DETECTRON2_DATASETS")
    assert detectron2_root is not None, 'Need to set $DETECTRON2_DATASETS enviroment variable!'

    train_set = ["0000",  "0001",  "0003",  "0004",  "0005",  "0009",  "0011",  "0012",  "0015",  "0017",  "0019",  "0020"]
    val_set = ["0002",  "0006",  "0007",  "0008",  "0010",  "0013",  "0014",  "0016",  "0018"]

    for vid in args.videos:
        split = "train" if vid in train_set else "val"
        input_label_dir = os.path.join(detectron2_root, "kitti_step/panoptic_maps",
                                        split, vid)
        in_dir = os.path.join(detectron2_root, "kitti_step/images", 
                                        split, vid)
        in_images = os.path.join(in_dir, "*.png")
        
        
        for base_lr in args.base_lrs:
                train_dir = os.path.join(detectron2_root, "kitti_step", "images", split, vid)
                val_dir = os.path.join(detectron2_root, "kitti_step", "panoptic_maps", split, vid)
                root_dir = os.path.join(args.output_dir, 
                                        vid + "_" + 
                                        str(base_lr) + "_" +
                                        str(args.batch_size) + "_" +
                                        str(args.optim) + "_" +
                                        str(args.update_every_n)
                                        )
                os.makedirs(root_dir, exist_ok=True)


                # 1. Create output folder and copy relevant data over from dataset root
                data_dir = os.path.join(root_dir, "data")
                os.makedirs(data_dir, exist_ok=True)
                data_train_dir = os.path.join(data_dir, "train")
                if os.path.exists(data_train_dir):
                    shutil.rmtree(data_train_dir)
                shutil.copytree(train_dir, data_train_dir)
                data_val_dir = os.path.join(data_dir, "val")
                if os.path.exists(data_val_dir):
                    shutil.rmtree(data_val_dir)
                shutil.copytree(val_dir, data_val_dir)

                exec_args = [sys.executable, "train_ttt_mae.py",
                    "--num-gpus", "1",
                    "--config-file", "configs/kitti_step/semantic-segmentation/swin/maskformer2_swin_small_bs16_90k_ttt_mae.yaml",
                    "--drop_aug",
                    "--ttt_in_dir", in_dir,
                    "--ttt_out_dir", data_dir,
                    "--exp_dir", "exp_dir/mae_ks_sema_" +
                            str(args.batch_size) + "_" + 
                            str(base_lr)
                ]
                resume_args = ["--resume"] if args.resume else []
                detectron_args = ["MODEL.WEIGHTS", args.weights,
                    "OUTPUT_DIR", root_dir,
                    "SOLVER.IMS_PER_BATCH", str(args.batch_size),
                    "SOLVER.ACCUM_ITER", str(args.accum_iter),
                    "SOLVER.UPDATE_EVERY_N", str(args.update_every_n),
                    "SOLVER.BASE_LR", str(base_lr),
                    "SOLVER.OPTIMIZER", str(args.optim),
                    "TTT.USE_SEG_HEAD", "False",
                    "TTT.RESTART_OPTIMIZER", 'True' if args.restart_optimizer else 'False',
                ]
                final_args = exec_args + resume_args + detectron_args

                subprocess.run(final_args, check=True)
    
                # Remove the appropriate directory after training
                if os.path.exists(root_dir):
                    shutil.rmtree(root_dir)
            


