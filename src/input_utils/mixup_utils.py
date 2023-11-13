""" Mixup and Cutmix

Papers:
mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)

CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features (https://arxiv.org/abs/1905.04899)

Code Reference:
CutMix: https://github.com/clovaai/CutMix-PyTorch

Hacked together by / Copyright 2019, Ross Wightman
"""
import numpy as np
import torch


def one_hot(x, num_classes, on_value=1.0, off_value=0.0, device="cpu"):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)


def mixup_target(target, num_classes, lam=1.0, smoothing=0.0, device="cpu", index=None):
    off_value = smoothing / num_classes
    on_value = 1.0 - smoothing + off_value
    y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value, device=device)
    target_sub = target.flip(0) if index is None else target[index]
    y2 = one_hot(target_sub, num_classes, on_value=on_value, off_value=off_value, device=device)
    return y1 * lam + y2 * (1.0 - lam)


def rand_bbox(img_shape, lam, margin=0.0, count=None):
    """Standard CutMix bounding-box
    Generates a random square bbox based on lambda value. This impl includes
    support for enforcing a border margin as percent of bbox dimensions.

    Args:
        img_shape (tuple): Image shape as tuple
        lam (float): Cutmix lambda value
        margin (float): Percentage of bbox dimension to enforce as margin (reduce amount of box outside image)
        count (int): Number of bbox to generate
    """
    ratio = np.sqrt(1 - lam)
    img_h, img_w = img_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
    cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
    cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
    yl = np.clip(cy - cut_h // 2, 0, img_h)
    yh = np.clip(cy + cut_h // 2, 0, img_h)
    xl = np.clip(cx - cut_w // 2, 0, img_w)
    xh = np.clip(cx + cut_w // 2, 0, img_w)
    return yl, yh, xl, xh


def rand_bbox_minmax(img_shape, minmax, count=None):
    """Min-Max CutMix bounding-box
    Inspired by Darknet cutmix impl, generates a random rectangular bbox
    based on min/max percent values applied to each dimension of the input image.

    Typical defaults for minmax are usually in the  .2-.3 for min and .8-.9 range for max.

    Args:
        img_shape (tuple): Image shape as tuple
        minmax (tuple or list): Min and max bbox ratios (as percent of image size)
        count (int): Number of bbox to generate
    """
    assert len(minmax) == 2
    img_h, img_w = img_shape[-2:]
    cut_h = np.random.randint(int(img_h * minmax[0]), int(img_h * minmax[1]), size=count)
    cut_w = np.random.randint(int(img_w * minmax[0]), int(img_w * minmax[1]), size=count)
    yl = np.random.randint(0, img_h - cut_h, size=count)
    xl = np.random.randint(0, img_w - cut_w, size=count)
    yu = yl + cut_h
    xu = xl + cut_w
    return yl, yu, xl, xu


def cutmix_bbox_and_lam(img_shape, lam, ratio_minmax=None, correct_lam=False, count=None):
    """Generate bbox and apply lambda correction."""
    if ratio_minmax is not None:
        yl, yu, xl, xu = rand_bbox_minmax(img_shape, ratio_minmax, count=count)
    else:
        yl, yu, xl, xu = rand_bbox(img_shape, lam, count=count)
    if correct_lam or ratio_minmax is not None:
        bbox_area = (yu - yl) * (xu - xl)
        lam = 1.0 - bbox_area / float(img_shape[-2] * img_shape[-1])
    return (yl, yu, xl, xu), lam


class Mixup:
    """Mixup/Cutmix that applies different params to each element or whole batch

    Args:
        mixup_alpha (float): mixup alpha value, mixup is active if > 0.
        cutmix_alpha (float): cutmix alpha value, cutmix is active if > 0.
        cutmix_minmax (List[float]): cutmix min/max image ratio, cutmix is active and uses this vs alpha if not None.
        prob (float): probability of applying mixup or cutmix per batch or element
        switch_prob (float): probability of switching to cutmix instead of mixup when both are active
        mode (str): how to apply mixup/cutmix params (per 'batch', 'pair' (pair of elements), 'elem' (element)
        correct_lam (bool): apply lambda correction when cutmix bbox clipped by image borders
        label_smoothing (float): apply label smoothing to the mixed target tensor
        num_classes (int): number of classes for target
    """

    def __init__(
        self,
        mixup_alpha=1.0,
        cutmix_alpha=0.0,
        cutmix_minmax=None,
        prob=1.0,
        switch_prob=0.5,
        mode="batch",
        correct_lam=False,
        label_smoothing=0.1,
        num_classes=1000,
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        if self.cutmix_minmax is not None:
            assert len(self.cutmix_minmax) == 2
            # force cutmix alpha == 1.0 when minmax active to keep logic simple & safe
            self.cutmix_alpha = 1.0
        self.mix_prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.mode = mode
        self.correct_lam = correct_lam  # correct lambda based on clipped area for cutmix
        self.mixup_enabled = True  # set to false to disable mixing (intended tp be set by train loop)

    def _params_per_elem(self, batch_size):
        """Original timm implementation + location and modality integration"""
        lam = np.ones(batch_size, dtype=np.float32)
        use_cutmix = np.zeros(batch_size, dtype=np.bool)
        if self.mixup_enabled:
            if self.mixup_alpha > 0.0 and self.cutmix_alpha > 0.0:
                use_cutmix = np.random.rand(batch_size) < self.switch_prob
                lam_mix = np.where(
                    use_cutmix,
                    np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size),
                    np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size),
                )
            elif self.mixup_alpha > 0.0:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size)
            elif self.cutmix_alpha > 0.0:
                use_cutmix = np.ones(batch_size, dtype=np.bool)
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = np.where(np.random.rand(batch_size) < self.mix_prob, lam_mix.astype(np.float32), lam)
        return lam, use_cutmix

    def _params_per_batch(self):
        """Original timm implementation + location and modality integration"""
        lam = 1.0
        use_cutmix = False
        if self.mixup_enabled and np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0.0 and self.cutmix_alpha > 0.0:
                use_cutmix = np.random.rand() < self.switch_prob
                lam_mix = (
                    np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
                    if use_cutmix
                    else np.random.beta(self.mixup_alpha, self.mixup_alpha)
                )
            elif self.mixup_alpha > 0.0:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.cutmix_alpha > 0.0:
                use_cutmix = True
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = float(lam_mix)
        return lam, use_cutmix

    def _mix_elem(self, x, args):
        """Original timm implementation + location and modality integration"""
        lam_batches = []
        for loc in args["location_names"]:
            for mod in args["modality_names"]:
                batch_size = len(x[loc][mod])
                lam_batch, use_cutmix = self._params_per_elem(batch_size)
                x_orig = x[loc][mod].clone()  # need to keep an unmodified original for mixing source
                for i in range(batch_size):
                    j = batch_size - i - 1
                    lam = lam_batch[i]
                    if lam != 1.0:
                        if use_cutmix[i]:
                            (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                                x[loc][mod][i].shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam
                            )
                            x[loc][mod][i][:, yl:yh, xl:xh] = x_orig[j][:, yl:yh, xl:xh]
                            lam_batch[i] = lam
                        else:
                            x[loc][mod][i] = x[loc][mod][i] * lam + x_orig[j] * (1 - lam)
                lam_batches.append(torch.tensor(lam_batch, device=x.device, dtype=x[loc][mod].dtype).unsqueeze(1))
        return torch.mean(lam_batches, axis=0), None

    def _mix_pair(self, x, args):
        """Original timm implementation + location and modality integration"""
        lam_batches = []
        for loc in args["location_names"]:
            for mod in args["modality_names"]:
                batch_size = len(x[loc][mod])
                lam_batch, use_cutmix = self._params_per_elem(batch_size // 2)
                x_orig = x[loc][mod].clone()  # need to keep an unmodified original for mixing source
                for i in range(batch_size // 2):
                    j = batch_size - i - 1
                    lam = lam_batch[i]
                    if lam != 1.0:
                        if use_cutmix[i]:
                            (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                                x[loc][mod][i].shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam
                            )
                            x[loc][mod][i][:, yl:yh, xl:xh] = x_orig[j][:, yl:yh, xl:xh]
                            x[loc][mod][j][:, yl:yh, xl:xh] = x_orig[i][:, yl:yh, xl:xh]
                            lam_batch[i] = lam
                        else:
                            x[loc][mod][i] = x[loc][mod][i] * lam + x_orig[j] * (1 - lam)
                            x[loc][mod][j] = x[loc][mod][j] * lam + x_orig[i] * (1 - lam)
                lam_batch = np.concatenate((lam_batch, lam_batch[::-1]))
                lam_batch = torch.tensor(lam_batch, device=x.device, dtype=x[loc][mod].dtype).unsqueeze(1)
                lam_batches.append(lam_batch)
        return torch.mean(lam_batches, axis=0), None

    def _mix_batch(self, x, args):
        """Original timm implementation + location and modality integration"""
        lam, use_cutmix = self._params_per_batch()
        lams = []
        if lam == 1.0:
            return 1.0, None
        if use_cutmix:
            for loc in args["location_names"]:
                for mod in args["modality_names"]:
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        x[loc][mod].shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam
                    )
                    x[loc][mod][:, :, yl:yh, xl:xh] = x[loc][mod].flip(0)[:, :, yl:yh, xl:xh]
                    lams.append(lam)
        else:
            for loc in args["location_names"]:
                for mod in args["modality_names"]:
                    x_flipped = x[loc][mod].flip(0).mul_(1.0 - lam)
                    x[loc][mod].mul_(lam).add_(x_flipped)
        return lam, None

    def _mix_batch_random(self, x, args):
        lam, use_cutmix = self._params_per_batch()
        if lam == 1:
            return 1.0, None

        rand_index = None
        if use_cutmix:
            # if we are using cutmix approach
            for loc in args["location_names"]:
                for mod in args["modality_names"]:
                    # use random index
                    if rand_index is None:
                        rand_index = torch.randperm(x[loc][mod].size()[0])
                    # get bounding box
                    (yl, yh, xl, xh) = rand_bbox(x[loc][mod].size(), lam)
                    x[loc][mod][:, :, yl:yh, xl:xh] = x[loc][mod][rand_index, :, yl:yh, xl:xh]
        else:
            # if we are using mixup
            for loc in args["location_names"]:
                for mod in args["modality_names"]:
                    # use random index
                    if rand_index is None:
                        rand_index = torch.randperm(x[loc][mod].size()[0])
                    x_flipped = x[loc][mod][rand_index].mul_(1.0 - lam)
                    x[loc][mod].mul_(lam).add_(x_flipped)
        return lam, rand_index

    def __call__(self, x, target, args):
        index = None
        if self.mode == "elem":
            lam, index = self._mix_elem(x, args)
        elif self.mode == "pair":
            lam, index = self._mix_pair(x, args)
        elif self.mode == "random_batch":
            lam, index = self._mix_batch_random(x, args)
        else:
            lam, index = self._mix_batch(x, args)
        target = mixup_target(target, self.num_classes, lam, self.label_smoothing, device=target.device, index=index)
        return x, target
