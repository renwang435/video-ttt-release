# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling
# config
from .config import (add_dropout_config,
                     add_maskformer2_config,
                     add_pretrain_config, add_ttt_config)
# dataset loading
from .data.dataset_mappers.instance_pretrain_dataset_mapper import \
    InstancePretrainDatasetMapper
from .data.dataset_mappers.panoptic_pretrain_dataset_mapper import \
    PanopticPretrainDatasetMapper
from .data.dataset_mappers.semantic_pretrain_dataset_mapper import (
    SemanticPretrainDatasetMapper)
# models
from .maskformer_ttt_model import MaskFormerTTT
from .test_time_augmentation import SemanticSegmentorWithTTA
