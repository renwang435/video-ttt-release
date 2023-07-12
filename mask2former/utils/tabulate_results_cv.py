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

    videos = ["bangkok", "bar", "berkeley", "havana", "house", "irvine", "paris", "restaurant", "school", "tokyo"]

    table = {
        "video" : [],
        "mIoU" : [],
    }
    vid_lengths = np.array([310, 196, 380, 325, 338, 340, 331, 305, 304, 301])
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
        print('Mean:', np.sum(vid_lengths * np.array(list(map(float, table['mIoU'][:len(vid_lengths)])))) / np.sum(vid_lengths))
    except:
        print('Error with mean.')

        

