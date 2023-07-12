try:
    # ignore ShapelyDeprecationWarning from fvcore
    import warnings

    from shapely.errors import ShapelyDeprecationWarning
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass
import contextlib
import copy
import io
import itertools
import logging
import os
import shutil
import time
import weakref
from collections import OrderedDict
from glob import glob
from typing import Any, Dict, List, Set

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import detectron2.utils.comm as comm
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (DatasetCatalog, MetadataCatalog,
                             build_detection_test_loader,
                             build_detection_train_loader)
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.engine.defaults import (AMPTrainer, SimpleTrainer, TrainerBase,
                                        create_ddp_model, default_writers,
                                        hooks)
from detectron2.evaluation import (DatasetEvaluator, DatasetEvaluators, SemSegEvaluator,
                                   print_csv_format, verify_results)
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from fvcore.nn.precise_bn import get_bn_modules
from torch.cuda.amp import autocast

# MaskFormer
from mask2former import (InstancePretrainDatasetMapper,
                         PanopticPretrainDatasetMapper,
                         SemanticPretrainDatasetMapper,
                         SemanticSegmentorWithTTA, add_dropout_config,
                         add_maskformer2_config, add_pretrain_config,
                         add_ttt_config)
from mask2former.data.dataloaders import MAEDataloader, MAETestPredictor

try:
    from mask2former.data.datasets.register_in_train_loop_coco_videos import (
        COCOVideos, register_custom_coco_videos)
    from mask2former.data.datasets.register_in_train_loop_coco_videos_panoptic import \
        register_custom_coco_videos_panoptic
except:
    print('COCO custom registration not found on this machine!')
from mask2former.data.datasets.register_in_train_loop_kitti import (
    get_parser, register_custom_kitti)
from mask2former.evaluation.coco_videos_instance_evaluation import \
    COCOVideosEvaluator
from mask2former.evaluation.coco_videos_panoptic_evaluation import \
    COCOVideosPanopticEvaluator


class Trainer(TrainerBase):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        ddp_unused = cfg.TTT.ST_ITERS is not None
        model = create_ddp_model(model, find_unused_parameters=ddp_unused, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration
            self.start_iter = self.iter + 1

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            # ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, 500))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def build_writers(self):
        """
        Build a list of writers to be used using :func:`default_writers()`.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.
        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        """
        return default_writers(self.cfg.OUTPUT_DIR, self.max_iter)

    def train(self):
        """
        Run training.
        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        super().train(self.start_iter, self.max_iter)
        if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
            assert hasattr(
                self, "_last_eval_results"
            ), "No evaluation results obtained during training!"
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def run_step(self):
        self._trainer.iter = self.iter
        assert self._trainer.model.training, "[Reimplemented AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[Reimplemented AMPTrainer] CUDA is required for AMP training!"

        # Check if we do loss update on this iteration
        if ((self.iter // self.cfg.TTT.ST_ITERS) + 1) % self.cfg.SOLVER.UPDATE_EVERY_N == 0:
            start = time.perf_counter()
            data = next(self._trainer._data_loader_iter) # this is a list with batch_size dicts that include 'file_name', 'height', 'width', 'image', 'image_mask'
            data_time = time.perf_counter() - start
            loss_dict_accum = {'loss_recon': 0}
            batch_size = len(data)
            accum_iter = self.cfg.SOLVER.ACCUM_ITER
            assert batch_size % accum_iter == 0 
            data_step = batch_size // accum_iter
            self._trainer.optimizer.zero_grad()
            for i in range(accum_iter):
                with autocast():
                    loss_dict = self._trainer.model(data[data_step *i:data_step * (i + 1)]) # A dict with {'loss_recon': ...}
                    losses = sum(loss_dict.values()) / accum_iter
                    
                    for key, value in loss_dict.items():
                        loss_dict_accum[key] += value / accum_iter

                self._trainer.grad_scaler.scale(losses).backward() # happens in each accum_iter

            self._trainer._write_metrics(loss_dict_accum, data_time)

            self._trainer.grad_scaler.step(self._trainer.optimizer) # happens only once
            self._trainer.grad_scaler.update()

    def state_dict(self):
        ret = super().state_dict()
        ret["_trainer"] = self._trainer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._trainer.load_state_dict(state_dict["_trainer"])

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:
        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []

        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # semantic segmentation
        if evaluator_type == "sem_seg":
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder,))
        
        if evaluator_type == "coco_videos_instance_seg":
            evaluator_list.append(COCOVideosEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_videos_panoptic_seg":
            evaluator_list.append(COCOVideosPanopticEvaluator(dataset_name, output_folder))
        
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg, queue=[0], it=0):
        if cfg.TTT.TASK == 'semantic-seg':
            for name in ["train", "val"]:
                DatasetCatalog.remove(f"kitti_step_video_sem_seg_{name}")
                MetadataCatalog.remove(f"kitti_step_video_sem_seg_{name}")
            cls.queue = queue
            register_custom_kitti(cfg.TTT.OUT_DIR, queue, it)
        elif cfg.TTT.TASK == 'instance-seg':
            for name in ["train", "val"]:
                DatasetCatalog.remove(f"coco_videos_instance_{name}")
                MetadataCatalog.remove(f"coco_videos_instance_{name}")
            cls.queue = queue
            register_custom_coco_videos(cls.coco_api, cls.all_raw_imgs, queue, it)
        elif cfg.TTT.TASK == 'panoptic-seg':
            for name in ["train", "val_with_sem_seg"]:
                DatasetCatalog.remove(f"coco_videos_panoptic_{name}")
                MetadataCatalog.remove(f"coco_videos_panoptic_{name}")
            cls.queue = queue
            register_custom_coco_videos_panoptic(cls.all_raw_imgs, queue, it)
        else:
            raise NotImplementedError

        if cfg.INPUT.DATASET_MAPPER_NAME == "pretrain_instance":
            mapper = InstancePretrainDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "pretrain_panoptic":
            mapper = PanopticPretrainDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        elif cfg.INPUT.DATASET_MAPPER_NAME == "pretrain_semantic":
            mapper = SemanticPretrainDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)


    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if cfg.TTT.TASK == 'semantic-seg':
            for name in ["train", "val"]:
                DatasetCatalog.remove(f"kitti_step_video_sem_seg_{name}")
                MetadataCatalog.remove(f"kitti_step_video_sem_seg_{name}")

            register_custom_kitti(cfg.TTT.OUT_DIR, cls.queue, 0)
        elif cfg.TTT.TASK == 'instance-seg':
            # Deal with insidious caching
            if os.path.exists(os.path.join(cfg.TTT.OUT_DIR.split('data')[0], 'inference')):
                shutil.rmtree(os.path.join(cfg.TTT.OUT_DIR.split('data')[0], 'inference'))

            for name in ["train", "val"]:
                DatasetCatalog.remove(f"coco_videos_instance_{name}")
                MetadataCatalog.remove(f"coco_videos_instance_{name}")

            register_custom_coco_videos(cls.coco_api, cls.all_raw_imgs, cls.queue, 0)
        elif cfg.TTT.TASK == 'panoptic-seg':
            # Deal with insidious caching
            if os.path.exists(os.path.join(cfg.TTT.OUT_DIR.split('data')[0], 'inference')):
                shutil.rmtree(os.path.join(cfg.TTT.OUT_DIR.split('data')[0], 'inference'))

            for name in ["train", "val_with_sem_seg"]:
                DatasetCatalog.remove(f"coco_videos_panoptic_{name}")
                MetadataCatalog.remove(f"coco_videos_panoptic_{name}")

            register_custom_coco_videos_panoptic(cls.all_raw_imgs, cls.queue, 0)
        else:
            raise NotImplementedError

        return build_detection_test_loader(cfg, dataset_name)


    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    def restart_optimizer(self):
        optimizer = self.build_optimizer(self.cfg, self._trainer.model)
        self._trainer.optimizer = optimizer
    
    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.
        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )
        
        save_dir = cfg.OUTPUT_DIR
        predictor = MAETestPredictor(cfg)

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = predictor.custom_inference_on_dataset(save_dir, model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    @staticmethod
    def auto_scale_workers(cfg, num_workers: int):
        """
        When the config is defined for certain number of workers (according to
        ``cfg.SOLVER.REFERENCE_WORLD_SIZE``) that's different from the number of
        workers currently in use, returns a new cfg where the total batch size
        is scaled so that the per-GPU batch size stays the same as the
        original ``IMS_PER_BATCH // REFERENCE_WORLD_SIZE``.
        Other config options are also scaled accordingly:
        * training steps and warmup steps are scaled inverse proportionally.
        * learning rate are scaled proportionally, following :paper:`ImageNet in 1h`.
        For example, with the original config like the following:
        .. code-block:: yaml
            IMS_PER_BATCH: 16
            BASE_LR: 0.1
            REFERENCE_WORLD_SIZE: 8
            MAX_ITER: 5000
            STEPS: (4000,)
            CHECKPOINT_PERIOD: 1000
        When this config is used on 16 GPUs instead of the reference number 8,
        calling this method will return a new config with:
        .. code-block:: yaml
            IMS_PER_BATCH: 32
            BASE_LR: 0.2
            REFERENCE_WORLD_SIZE: 16
            MAX_ITER: 2500
            STEPS: (2000,)
            CHECKPOINT_PERIOD: 500
        Note that both the original config and this new config can be trained on 16 GPUs.
        It's up to user whether to enable this feature (by setting ``REFERENCE_WORLD_SIZE``).
        Returns:
            CfgNode: a new config. Same as original if ``cfg.SOLVER.REFERENCE_WORLD_SIZE==0``.
        """
        old_world_size = cfg.SOLVER.REFERENCE_WORLD_SIZE
        if old_world_size == 0 or old_world_size == num_workers:
            return cfg
        cfg = cfg.clone()
        frozen = cfg.is_frozen()
        cfg.defrost()

        assert (
            cfg.SOLVER.IMS_PER_BATCH % old_world_size == 0
        ), "Invalid REFERENCE_WORLD_SIZE in config!"
        scale = num_workers / old_world_size
        bs = cfg.SOLVER.IMS_PER_BATCH = int(round(cfg.SOLVER.IMS_PER_BATCH * scale))
        lr = cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR * scale
        max_iter = cfg.SOLVER.MAX_ITER = int(round(cfg.SOLVER.MAX_ITER / scale))
        warmup_iter = cfg.SOLVER.WARMUP_ITERS = int(round(cfg.SOLVER.WARMUP_ITERS / scale))
        cfg.SOLVER.STEPS = tuple(int(round(s / scale)) for s in cfg.SOLVER.STEPS)
        cfg.TEST.EVAL_PERIOD = int(round(cfg.TEST.EVAL_PERIOD / scale))
        cfg.SOLVER.CHECKPOINT_PERIOD = int(round(cfg.SOLVER.CHECKPOINT_PERIOD / scale))
        cfg.SOLVER.REFERENCE_WORLD_SIZE = num_workers  # maintain invariant
        logger = logging.getLogger(__name__)
        logger.info(
            f"Auto-scaling the config to batch_size={bs}, learning_rate={lr}, "
            f"max_iter={max_iter}, warmup={warmup_iter}."
        )

        if frozen:
            cfg.freeze()
        return cfg


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_ttt_config(cfg)
    add_dropout_config(cfg)
    add_pretrain_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.TTT.IN_DIR = args.ttt_in_dir
    cfg.TTT.OUT_DIR = args.ttt_out_dir
    cfg.TTT.EXP_DIR = args.exp_dir

    cfg.DROPOUT_AUG.ENABLED = args.drop_aug

    # Masking options for MAE
    cfg.PRETRAIN.MASK_TYPE = "const"

    if "semantic-segmentation" in args.config_file:
        cfg.TTT.TASK = 'semantic-seg'
    elif "instance-segmentation" in args.config_file:
        cfg.TTT.TASK = 'instance-seg'
    elif "panoptic-segmentation" in args.config_file:
        cfg.TTT.TASK = 'panoptic-seg'
    else:
        raise NotImplementedError

    cfg.TTT.REENTRANT = args.resume

    for _attr in ["model", "data_loader", "optimizer"]:
            setattr(
                Trainer,
                _attr,
                property(
                    # getter
                    lambda self, x=_attr: getattr(self._trainer, x),
                    # setter
                    lambda self, value, x=_attr: setattr(self._trainer, x, value),
                ),
            )
    
    # Make master list of images for instance
    setattr(
        Trainer,
        "all_raw_imgs",
        sorted(glob(os.path.join(cfg.TTT.IN_DIR,"*.png")))
    )

    # Make master COCO api
    if cfg.TTT.COCO_VID is not None:
        _root = os.getenv("DETECTRON2_DATASETS")
        root = os.path.join(_root, "coco_videos")
        json_file_path = os.path.join(root, 'annotations', 'train', cfg.TTT.COCO_VID + '.json')
        json_file = PathManager.get_local_path(json_file_path)
        with contextlib.redirect_stdout(io.StringIO()):
            coco_api = COCOVideos(json_file)
        setattr(
            Trainer,
            "coco_api",
            coco_api
        )


    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")

    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )

        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    if cfg.TTT.TASK == 'semantic-seg':
        trainer.register_hooks(
            [MAEDataloader(register_custom_kitti, cfg)]
        )
    elif cfg.TTT.TASK == 'instance-seg':
        trainer.register_hooks(
            [MAEDataloader(register_custom_coco_videos, cfg)]
        )
    elif cfg.TTT.TASK == 'panoptic-seg':
        trainer.register_hooks(
            [MAEDataloader(register_custom_coco_videos_panoptic, cfg)]
        )
    else:
        raise NotImplementedError

    return trainer.train()


if __name__ == "__main__":
    args = get_parser().parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
