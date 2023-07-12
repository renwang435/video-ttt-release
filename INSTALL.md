## Installation

Our installation process follows heavily from that of [Mask2Former](https://github.com/facebookresearch/Mask2Former/blob/main/INSTALL.md). There are a few crucial differences:

1. We use an older fork of Detectron2 to ensure consistent results in the evaluation metrics.

2. We require that the environment variable `DETECTRON2_DATASETS` be set.

### Example conda environment setup
```bash
conda create --name vttt python=3.8 -y
conda activate vttt
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
pip install -U opencv-python

# under your working directory
git clone git@github.com:renwang435/detectron2.git
cd detectron2
pip install -e .
pip install git+https://github.com/cocodataset/panopticapi.git

cd ..
git clone git@github.com:renwang435/video-ttt-release.git
cd video-ttt-release
pip install -r requirements.txt
cd mask2former/modeling/pixel_decoder/ops
sh make.sh

echo "export DETECTRON2_DATASETS=/path/to/datasets" >> ~/.bashrc
```
