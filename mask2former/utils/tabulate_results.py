import argparse
import json
import os
import os.path as osp
from collections import OrderedDict
from copy import deepcopy
from glob import glob

import numpy as np
from tabulate import tabulate

parser = argparse.ArgumentParser()
parser.add_argument(
    '--root_dir',
    type=str,
    default=None
)
args = parser.parse_args()

if __name__ == '__main__':
    root_dir = args.root_dir
    exps = glob(root_dir + "/*/", recursive=False)

    train_set = ["0000",  "0001",  "0003",  "0004",  "0005",  "0009",  "0011",  "0012",  "0015",  "0017",  "0019",  "0020"]
    val_set = ["0002",  "0006",  "0007",  "0008",  "0010",  "0013",  "0014",  "0016",  "0018"]

    videos = deepcopy(train_set)
    videos.extend(val_set)

    table = {
        "video" : [],
        "mIoU" : [],
    }
    train_length = np.array([154, 447, 144, 314, 297, 803, 373, 78, 476, 145, 1059, 837])
    val_length = np.array([233, 270, 800, 390, 294, 340, 106, 209, 339])
    for vid in videos:
        try:
            # import ipdb; ipdb.set_trace()
            results = [p for p in exps if "/" + vid in p]
            assert len(results) == 1, "Expected 1 video, got " + str(len(results)) + " videos"
            fp = open(os.path.join(results[0], "16_win", 'performance.txt'), 'r')
            out = fp.readlines()
            miou = np.mean([float(run) for run in out])
            table["video"].append(vid)
            table["mIoU"].append(str(miou))
            fp.close()
        except:
            table["video"].append(vid)
            table["mIoU"].append("")
        
    # import ipdb; ipdb.set_trace()
    table = OrderedDict(sorted(table.items()))
    print(tabulate(table, headers="keys", tablefmt="simple"))
    print()
    try:
        print('Train Mean:', np.sum(train_length * np.array(list(map(float, table['mIoU'][:len(train_length)])))) / np.sum(train_length))
    except:
        print('Error with training mean.')
    try:
        print('Val Mean:', np.sum(val_length * np.array(list(map(float, table['mIoU'][-len(val_length):])))) / np.sum(val_length))
    except:
        print('Error with val mean.')

        

