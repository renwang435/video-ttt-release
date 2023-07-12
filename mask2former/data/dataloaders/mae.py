import datetime
import logging
import os
import sys
import time
from collections import abc
from contextlib import ExitStack
from glob import glob
from typing import List, Union

import detectron2.data.transforms as T
import numpy as np
import torch
import torch.nn as nn
from detectron2.data import MetadataCatalog
from detectron2.engine import HookBase
from detectron2.engine.defaults import DefaultPredictor
from detectron2.evaluation import (DatasetEvaluator, DatasetEvaluators,
                                   inference_context)
from detectron2.structures import ImageList
from detectron2.utils.comm import get_world_size
from detectron2.utils.logger import log_every_n_seconds
from detectron2.utils.visualizer import (ColorMode, GenericMask, Visualizer,
                                         _PanopticPrediction)
from PIL import Image

_OFF_WHITE = (1.0, 1.0, 240.0 / 255)

class CustomVisualizer(Visualizer):
    def __init__(self, img_rgb, metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE):
        super().__init__(img_rgb, metadata, scale, instance_mode)

    def draw_instance_predictions(self, predictions):
        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
        # labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None
        
        # import ipdb; ipdb.set_trace()

        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            colors = [
                # self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
                [x / 255 for x in self.metadata.thing_colors[c]] for c in classes
            ]
            alpha = 0.8
        else:
            colors = None
            alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.reset_image(
                self._create_grayscale_image(
                    (predictions.pred_masks.any(dim=0) > 0).numpy()
                    if predictions.has("pred_masks")
                    else None
                )
            )
            alpha = 0.3

        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            # labels=labels,
            labels=None,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output

    def draw_panoptic_seg_predictions(self, panoptic_seg, segments_info, area_threshold=None, alpha=0.7):
        """
        Draw panoptic prediction annotations or results.

        Args:
            panoptic_seg (Tensor): of shape (height, width) where the values are ids for each
                segment.
            segments_info (list[dict] or None): Describe each segment in `panoptic_seg`.
                If it is a ``list[dict]``, each dict contains keys "id", "category_id".
                If None, category id of each pixel is computed by
                ``pixel // metadata.label_divisor``.
            area_threshold (int): stuff segments with less than `area_threshold` are not drawn.

        Returns:
            output (VisImage): image object with visualizations.
        """
        pred = _PanopticPrediction(panoptic_seg, segments_info, self.metadata)

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.reset_image(self._create_grayscale_image(pred.non_empty_mask()))

        # draw mask for all semantic segments first i.e. "stuff"
        for mask, sinfo in pred.semantic_masks():
            category_idx = sinfo["category_id"]
            try:
                mask_color = [x / 255 for x in self.metadata.stuff_colors[category_idx]]
            except AttributeError:
                mask_color = None

            # text = self.metadata.stuff_classes[category_idx]
            self.draw_binary_mask(
                mask,
                color=mask_color,
                edge_color=_OFF_WHITE,
                # text=text,
                alpha=alpha,
                area_threshold=area_threshold,
            )

        # draw mask for all instances second
        all_instances = list(pred.instance_masks())
        if len(all_instances) == 0:
            return self.output
        masks, sinfo = list(zip(*all_instances))
        category_ids = [x["category_id"] for x in sinfo]

        try:
            scores = [x["score"] for x in sinfo]
        except KeyError:
            scores = None
        # labels = _create_text_labels(
        #     category_ids, scores, self.metadata.thing_classes, [x.get("iscrowd", 0) for x in sinfo]
        # )

        try:
            colors = [
                # self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in category_ids
                [x / 255 for x in self.metadata.thing_colors[c]] for c in category_ids
            ]
        except AttributeError:
            colors = None
        self.overlay_instances(
            masks=masks, 
            # labels=labels,
            labels=None,
            assigned_colors=colors, 
            alpha=alpha
        )

        # import ipdb; ipdb.set_trace()

        # self.overlay_instances(masks=masks, assigned_colors=colors, alpha=alpha)

        return self.output

    def draw_sem_seg(self, sem_seg, area_threshold=None, alpha=0.8):
        """
        Draw semantic segmentation predictions/labels.

        Args:
            sem_seg (Tensor or ndarray): the segmentation of shape (H, W).
                Each value is the integer label of the pixel.
            area_threshold (int): segments with less than `area_threshold` are not drawn.
            alpha (float): the larger it is, the more opaque the segmentations are.

        Returns:
            output (VisImage): image object with visualizations.
        """
        if isinstance(sem_seg, torch.Tensor):
            sem_seg = sem_seg.numpy()
        labels, areas = np.unique(sem_seg, return_counts=True)
        sorted_idxs = np.argsort(-areas).tolist()
        labels = labels[sorted_idxs]
        for label in filter(lambda l: l < len(self.metadata.stuff_classes), labels):
            try:
                mask_color = [x / 255 for x in self.metadata.stuff_colors[label]]
            except (AttributeError, IndexError):
                mask_color = None

            binary_mask = (sem_seg == label).astype(np.uint8)
            # text = self.metadata.stuff_classes[label]
            self.draw_binary_mask(
                binary_mask,
                color=mask_color,
                edge_color=_OFF_WHITE,
                # text=text,
                alpha=alpha,
                area_threshold=area_threshold,
            )
        return self.output



class MAEPredictor(DefaultPredictor):
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        
        self.mask_ratio = cfg.PRETRAIN.MASK_RATIO[0]
        self.patch_size = 16
        self.size_divisibility = cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY

        self.ttt_task = cfg.TTT.TASK

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], 
            cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format


    def __call__(self, original_image, model):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            images = [x for x in image]
            images = ImageList.from_tensors(images, self.size_divisibility)
            images = images.tensor

            # Generate random mask
            _, h, w = images.shape
            patched_img = self.patchify(images)

            _, seq_mask, _ = self.random_masking(patched_img, self.mask_ratio)

            mask = seq_mask.unsqueeze(-1).repeat_interleave(self.patch_size ** 2, dim=-1)
            unpatched_mask = self.unpatchify_mask(mask, h, w)

            inputs = {"image": image, "height": height, "width": width, "image_mask" : unpatched_mask}

            # inputs = {"image": image, "height": height, "width": width}
            predictions = model([inputs])[0]

            return predictions
    
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        imgs = imgs.unsqueeze(0)
        p = self.patch_size
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        
        return x
    
    def unpatchify(self, x, h, w):
        """
        x: (N, L, patch_size**2 * 3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        assert (h // p) * (w // p) == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h // p, w // p, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h, w))

        return imgs.squeeze(0)
    
    def unpatchify_mask(self, x, h, w):
        """
        x: (N, L, patch_size**2 * 1)
        imgs: (N, 1, H, W)
        """
        p = self.patch_size
        assert (h // p) * (w // p) == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h // p, w // p, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h, w))

        return imgs.squeeze(0)
 
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 1 is keep, 0 is remove
        mask = torch.zeros([N, L], device=x.device)
        mask[:, :len_keep] = 1
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

       
    def draw_predictions(self, original_image, preds):
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        if self.input_format == "BGR":
            # whether the model expects BGR inputs or RGB
            original_image = original_image[:, :, ::-1]
        
        # visualizer = Visualizer(original_image, self.metadata, instance_mode=ColorMode.SEGMENTATION)
        visualizer = CustomVisualizer(original_image, self.metadata, instance_mode=ColorMode.SEGMENTATION)
        if self.ttt_task == 'semantic-seg':
            vis_output = visualizer.draw_sem_seg(
                preds["sem_seg"].argmax(dim=0).cpu()
            )
        elif self.ttt_task == 'instance-seg':
            vis_output = visualizer.draw_instance_predictions(preds["instances"].to('cpu'))
        elif self.ttt_task == 'panoptic-seg':
            panoptic_seg, segments_info = preds["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.cpu(), segments_info
            )
            vis_output = visualizer.draw_instance_predictions(preds["instances"].to('cpu'))
        else:
            raise NotImplementedError

        return vis_output
    
    def draw_reconstructions(self, recon):
        # import ipdb; ipdb.set_trace()

        recon = Image.fromarray(recon.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8))

        return recon


class MAETestPredictor(MAEPredictor):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)


    def custom_inference_on_dataset(
        self,
        save_dir,
        model, data_loader,
        evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None]
    ):
        """
        Run model on the data_loader and evaluate the metrics with evaluator.
        Also benchmark the inference speed of `model.__call__` accurately.
        The model will be used in eval mode.
        Args:
            model (callable): a callable which takes an object from
                `data_loader` and returns some outputs.
                If it's an nn.Module, it will be temporarily set to `eval` mode.
                If you wish to evaluate a model in `training` mode instead, you can
                wrap the given model and override its behavior of `.eval()` and `.train()`.
            data_loader: an iterable object with a length.
                The elements it generates will be the inputs to the model.
            evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
                but don't want to do any evaluation.
        Returns:
            The return value of `evaluator.evaluate()`
        """
        num_devices = get_world_size()
        logger = logging.getLogger(__name__)
        logger.info("Start inference on {} batches".format(len(data_loader)))

        total = len(data_loader)  # inference data loader must have a fixed length
        if evaluator is None:
            # create a no-op evaluator
            evaluator = DatasetEvaluators([])
        if isinstance(evaluator, abc.MutableSequence):
            evaluator = DatasetEvaluators(evaluator)
        evaluator.reset()

        num_warmup = min(5, total - 1)
        start_time = time.perf_counter()
        total_data_time = 0
        total_compute_time = 0
        total_eval_time = 0
        with ExitStack() as stack:
            if isinstance(model, nn.Module):
                stack.enter_context(inference_context(model))
            stack.enter_context(torch.no_grad())

            start_data_time = time.perf_counter()
            for idx, inputs in enumerate(data_loader):
                assert len(inputs) == 1, 'Can only use this inference code for 1 image at a time'

                # import ipdb; ipdb.set_trace()

                image = inputs[0]['image']
                images = [x for x in image]
                images = ImageList.from_tensors(images, self.size_divisibility)
                images = images.tensor

                # Generate random mask
                _, h, w = images.shape
                patched_img = self.patchify(images)

                _, seq_mask, _ = self.random_masking(patched_img, self.mask_ratio)

                mask = seq_mask.unsqueeze(-1).repeat_interleave(self.patch_size ** 2, dim=-1)
                unpatched_mask = self.unpatchify_mask(mask, h, w)

                inputs[0]["image_mask"] = unpatched_mask

                total_data_time += time.perf_counter() - start_data_time
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_data_time = 0
                    total_compute_time = 0
                    total_eval_time = 0

                start_compute_time = time.perf_counter()
                outputs = model(inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_compute_time += time.perf_counter() - start_compute_time

                start_eval_time = time.perf_counter()
                evaluator.process(inputs, outputs)
                total_eval_time += time.perf_counter() - start_eval_time

                iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                data_seconds_per_iter = total_data_time / iters_after_start
                compute_seconds_per_iter = total_compute_time / iters_after_start
                eval_seconds_per_iter = total_eval_time / iters_after_start
                total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
                if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                    eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                    log_every_n_seconds(
                        logging.INFO,
                        (
                            f"Inference done {idx + 1}/{total}. "
                            f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                            f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                            f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                            f"Total: {total_seconds_per_iter:.4f} s/iter. "
                            f"ETA={eta}"
                        ),
                        n=5,
                    )
                start_data_time = time.perf_counter()

        # Measure the time only for this worker (before the synchronization barrier)
        total_time = time.perf_counter() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        # NOTE this format is parsed by grep
        logger.info(
            "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
                total_time_str, total_time / (total - num_warmup), num_devices
            )
        )
        total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
        logger.info(
            "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
                total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
            )
        )

        # import ipdb; ipdb.set_trace()

        results = evaluator.evaluate()
        # An evaluator may return None when not in main process.
        # Replace it by an empty dict instead to make it easier for downstream code to handle
        if results is None:
            results = {}

        return results



class MAEDataloader(HookBase):
    def __init__(self, registration, cfg):
        """
        Each argument is a function that takes one argument: the trainer.
        """
        self.registration = registration
        self.cfg = cfg.clone()
        win_size = cfg.TTT.WIN_SIZE
        self.win_size = "inf" if win_size == "inf" else int(win_size)
        self.predictor = MAEPredictor(cfg)
        self.restart_optimizer = cfg.TTT.RESTART_OPTIMIZER 
        self.ttt_in = cfg.TTT.IN_DIR
        self.video_dir = cfg.TTT.IN_DIR.split('/')[-1]
        self.ttt_out = os.path.join(cfg.TTT.OUT_DIR, "train")
        self.exp_dir = cfg.TTT.EXP_DIR
        os.makedirs(self.exp_dir, exist_ok=True)

        self.ttt_task = cfg.TTT.TASK
        self.coco_vid = cfg.TTT.COCO_VID
        if (self.ttt_task == 'instance-seg' or self.ttt_task == 'panoptic-seg') and self.coco_vid is not None:
            # Define the list of img_idxs for validation
            self.validation_img_idxs = np.arange(0, 4000, 10)
        else:
            self.validation_img_idxs = None
        
        # Master list of images
        self.all_raw_imgs = sorted(glob(os.path.join(self.ttt_in,"*.png")))

        # Make the log/save directory for this experiment
        self.mask_type = cfg.DROPOUT_AUG.MASK_TYPE
        self.mratio = str(cfg.DROPOUT_AUG.RATIO)
        self.exp_dir = os.path.join(self.exp_dir,
                                    str(self.video_dir) + "_" + self.mask_type + "_mask" + self.mratio
                                    )
        os.makedirs(self.exp_dir, exist_ok=True)
        self.exp_dir = os.path.join(self.exp_dir, str(self.win_size) + "_win")
        os.makedirs(self.exp_dir, exist_ok=True)

        if cfg.TTT.REENTRANT:
            assert cfg.TTT.CHECKPOINT_ITERS > 0, 'reentrant needs to start from most recent checkpoint'
            # Read log (must exist!)
            exp_log = os.path.join(self.exp_dir, 'performance.txt')
            with open(exp_log, 'r') as fp:
                prev_results = fp.readlines()
            
            # Edit existing log
            last_image = int(cfg.TTT.CHECKPOINT_ITERS) // int(cfg.TTT.ST_ITERS)
            keep_lines = last_image // 10 - 1
            try:
                assert keep_lines > 0, 'got keep_lines = {keep_lines} < 1, starting from scratch'
            except:
                # Remove log if exists
                keep_lines = 0
                if os.path.isfile(exp_log):
                    os.remove(exp_log)

            rewrite_results = prev_results[:keep_lines]
            with open(exp_log, 'w') as fp:
                for line in rewrite_results:
                    fp.write(line)
            
            self.imgs_in_queue = np.arange(last_image - self.win_size + 2, last_image + 2).tolist()
            self.internal_iter = 0
            self.max_st_iters = cfg.TTT.ST_ITERS
            self.max_img_idx = len(glob(os.path.join(self.ttt_in, "*.png")))
        else:
            # Remove log if exists
            exp_log = os.path.join(self.exp_dir, 'performance.txt')
            if os.path.isfile(exp_log):
                os.remove(exp_log)
            
            self.imgs_in_queue = [0]
            self.internal_iter = 0
            self.max_st_iters = cfg.TTT.ST_ITERS
            self.max_img_idx = len(glob(os.path.join(self.ttt_in, "*.png")))

        # Setting
        self.ttt_setting = cfg.TTT.SETTING
        self.orig_model_weights = cfg.MODEL.WEIGHTS

        self._root = os.getenv("DETECTRON2_DATASETS")


    # Re-register dataloader
    def before_step(self):
        if self.internal_iter == 0:
            self.trainer._trainer._data_loader_iter = iter(self.trainer.build_train_loader(self.cfg,
                                                                                self.imgs_in_queue,
                                                                                self.internal_iter))
        self.trainer.model.train()

        

    # Update model predictions on this image
    def after_step(self):
        if ((self.internal_iter + 1) == self.max_st_iters):
            if self.validation_img_idxs is None or self.imgs_in_queue[-1] in self.validation_img_idxs:
                # Just log from the beginning
                res = self.trainer.test(self.cfg, self.trainer.model)

                # Log results to txt file
                if self.ttt_task == 'semantic-seg':
                    try:
                        metric = str(res['sem_seg']['mIoU'])
                    except:
                        metric = "0.0"
                elif self.ttt_task == 'instance-seg':
                    try:
                        metric = str(res['segm']['AP'])
                    except:
                        metric = "0.0"
                elif self.ttt_task == 'panoptic-seg':
                    try:
                        metric = str(res['panoptic_seg']['PQ'])
                    except:
                        metric = "0.0"
                else:
                    raise NotImplementedError
            
                # Look for experiment log and append
                exp_log = os.path.join(self.exp_dir, 'performance.txt')
                with open(exp_log, 'a') as fp:
                    fp.write(metric + '\n')

            # Modify queue as necessary
            self.imgs_in_queue.append(self.imgs_in_queue[-1] + 1)
            # We will take one more image on the next iteration
            if self.win_size != "inf" and len(self.imgs_in_queue) > self.win_size:
                # Evict
                self.imgs_in_queue.pop(0)
            self.internal_iter = 0

            if self.ttt_setting == "standard":
                self.trainer.checkpointer.load(self.orig_model_weights)
            if self.restart_optimizer:
                self.trainer.restart_optimizer()
        else:
            self.internal_iter = self.internal_iter + 1
        
        if self.imgs_in_queue[-1] == self.max_img_idx:
            # Perform after_train and then exit
            self.after_train()
            sys.exit(0)


    def after_train(self):
        # Copy models
        self.trainer.checkpointer.save("model_final_" +
                                            str(self.video_dir) + "_" + self.mask_type  + "_mask" + self.mratio
                                      )
