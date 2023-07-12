# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
from fvcore.transforms.transform import Transform


class DropoutAug(Transform):
    def __init__(
        self,
        drop_ratio=0.5,
        mask_type="",
    ):
        super().__init__()

        # Random dropout or confidence-weighted dropout
        self.drop_ratio = drop_ratio
        self.mask_type = mask_type
        self.patch_size = 16

        self._set_attributes(locals())

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

    def apply_image(self, img, prev_conf_map, interp=None):
        if self.mask_type == 'rand':
            _, h, w = img.shape
            patched_img = self.patchify(img)

            _, mask, _ = self.random_masking(patched_img, self.drop_ratio)
            masked_img = patched_img * mask.unsqueeze(-1)

            masked_img = self.unpatchify(masked_img, h, w)
        
        elif self.mask_type == 'conf':
            masked_img = self.confidence_masking(img, prev_conf_map, self.drop_ratio)
        else:
            raise NotImplementedError

        return masked_img.type(torch.uint8)

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
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        # h = w = int(x.shape[1]**.5)
        assert (h // p) * (w // p) == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h // p, w // p, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h, w))

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

    def confidence_masking(self, x, prev_conf_map, mask_ratio):
        assert prev_conf_map is not None, 'Need a confidence map if dropout is not random'

        # Drop out most confident `mask_ratio` pixels
        rows, cols = prev_conf_map.shape
        topl = int((rows * cols) * mask_ratio)
        best_confs = torch.argsort(prev_conf_map.flatten(), descending=True)[:topl]
        row_idx = best_confs // cols
        col_idx = best_confs % cols
        mask = torch.ones_like(prev_conf_map, device=x.device)
        mask[row_idx, col_idx] = 0

        x = x * mask.unsqueeze(0)

        return x


class DropoutConcatAug(Transform):
    def __init__(
        self,
        drop_ratio=0.5,
        drop_type="",
        ignore_label=255,
    ):
        super().__init__()

        # Random or fixed
        if drop_type == 'const':
            assert len(drop_ratio) == 1, 'Fixed mask ratio ==> only 1 field in --mask-ratio'
        elif drop_type == 'rand':
            assert len(drop_ratio) == 2, 'Dynamic mask ratio ==> lower and upper bound in --mask-ratio'
        else:
            raise NotImplementedError

        self.drop_ratio = drop_ratio
        self.drop_type = drop_type
        self.patch_size = 16

        assert isinstance(ignore_label, int), 'ignore_label should be integer'
        self.ignore_label = ignore_label

        self._set_attributes(locals())

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, mask, segmentation):
        segmentation[mask.bool()] = self.ignore_label

        return segmentation

    def apply_image(self, img, interp=None):
        if self.drop_type == 'rand':
            drop_to_use = np.random.uniform(low=self.drop_ratio[0], high=self.drop_ratio[1])
        elif self.drop_type == 'const':
            drop_to_use = self.drop_ratio[0]
        else:
            raise NotImplementedError
        
        _, h, w = img.shape
        patched_img = self.patchify(img)

        _, seq_mask, _ = self.random_masking(patched_img, drop_to_use)

        # Concatenate mask with image
        # mask (N, L) --> mask (N, L, D)
        # patched_img --> (N, L, D)
        mask = seq_mask.unsqueeze(-1).repeat_interleave(self.patch_size ** 2, dim=-1)

        unpatched_mask = self.unpatchify_mask(mask, h, w)

        return unpatched_mask

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


class DropoutConcatAugFixed(Transform):
    def __init__(
        self,
        drop_ratio=0.5,
        mask_type="",
        ignore_label=255,
    ):
        super().__init__()

        # Random dropout or confidence-weighted dropout
        self.drop_ratio = drop_ratio
        self.mask_type = mask_type
        self.patch_size = 16

        assert isinstance(ignore_label, int), 'ignore_label should be integer'
        self.ignore_label = ignore_label

        self._set_attributes(locals())

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, mask, segmentation):

        return segmentation

    def apply_image(self, img, prev_conf_map, interp=None):
        if self.mask_type == 'rand':
            _, h, w = img.shape
            patched_img = self.patchify(img)

            _, mask, _ = self.random_masking(patched_img, self.drop_ratio)

            # Concatenate mask with image
            # mask (N, L) --> mask (N, L, D)
            # patched_img --> (N, L, D)
            mask = mask.unsqueeze(-1).repeat_interleave(self.patch_size ** 2, dim=-1)

            # import ipdb; ipdb.set_trace()

            unpatched_mask = self.unpatchify_mask(mask, h, w)
            input_img = img.float() * unpatched_mask
            masked_img = torch.cat([input_img, unpatched_mask], dim=0)
            
        else:
            raise NotImplementedError

        return masked_img.type(torch.uint8), unpatched_mask, img.float()

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
