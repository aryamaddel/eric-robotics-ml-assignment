"""Make Ultralytics YOLO26 blocks prunable by structured (channel) pruning.

C2f/C3k2 and C2PSA compute `cv1(x)` and then split its output channels in two (chunk/split).
Dependency-graph pruners mis-track channel indices across that split, so each block is replaced
by an equivalent split-free module whose `cv1` is two convolutions holding the corresponding
halves of the original weights and BatchNorm statistics. Outputs are bit-identical before pruning.

These are real classes (not monkey-patched forwards) so that pruned checkpoints can be pickled
by the Ultralytics trainer and loaded again for export.
"""

import copy

import torch
import torch.nn as nn

ROUTING_ATTRS = ("f", "i", "type", "np")  # attributes Ultralytics uses to route layers


def split_conv(conv, c):
    """Split an Ultralytics Conv (conv + bn + act) into two Convs over output channels [:c] and [c:]."""
    parts = []
    for sl in (slice(0, c), slice(c, None)):
        part = copy.deepcopy(conv)
        w = conv.conv.weight.data[sl].clone()
        part.conv = nn.Conv2d(conv.conv.in_channels, w.shape[0], conv.conv.kernel_size, conv.conv.stride,
                              conv.conv.padding, conv.conv.dilation, conv.conv.groups, bias=conv.conv.bias is not None)
        part.conv.weight.data = w
        if conv.conv.bias is not None:
            part.conv.bias.data = conv.conv.bias.data[sl].clone()
        bn = conv.bn
        part.bn = nn.BatchNorm2d(w.shape[0], eps=bn.eps, momentum=bn.momentum)
        for name in ("weight", "bias", "running_mean", "running_var"):
            getattr(part.bn, name).data = getattr(bn, name).data[sl].clone()
        part.bn.train(bn.training)  # a fresh BatchNorm starts in train mode (batch statistics)
        parts.append(part)
    return parts


class _SplitFree(nn.Module):
    def __init__(self, src):
        super().__init__()
        self.c = src.c
        self.cv1a, self.cv1b = split_conv(src.cv1, src.c)
        self.cv2, self.m = src.cv2, src.m
        for k in ROUTING_ATTRS:
            if hasattr(src, k):
                setattr(self, k, getattr(src, k))
        self.train(src.training)


class SplitFreeC2f(_SplitFree):
    """Equivalent of C2f / C3k2 without chunk()."""

    def forward(self, x):
        y = [self.cv1a(x), self.cv1b(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SplitFreeC2PSA(_SplitFree):
    """Equivalent of C2PSA without split()."""

    def forward(self, x):
        return self.cv2(torch.cat((self.cv1a(x), self.m(self.cv1b(x))), 1))


def remove_splits(model) -> int:
    """Replace top-level C2f/C3k2/C2PSA layers of a DetectionModel in place; returns the count."""
    from ultralytics.nn.modules.block import C2f, C2PSA

    n = 0
    for idx, layer in enumerate(model.model):
        if isinstance(layer, C2PSA):
            model.model[idx] = SplitFreeC2PSA(layer)
        elif isinstance(layer, C2f):
            model.model[idx] = SplitFreeC2f(layer)
        else:
            continue
        n += 1
    return n
