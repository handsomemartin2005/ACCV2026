# Ultralytics YOLO 🚀, AGPL-3.0 license
import timm
import torch
import torch.nn as nn

from ultralytics.nn.modules import *

from ultralytics.utils.torch_utils import (fuse_conv_and_bn, fuse_deconv_and_bn,model_info)


try:
    import thop
except ImportError:
    thop = None


def is_fused(thresh=10, model=None):
    """
    Check if the model has less than a certain threshold of BatchNorm layers.
    检查模型的 BatchNorm 层数是否少于某个阈值。

    Args:
        thresh (int, optional): The threshold number of BatchNorm layers. Default is 10.

    Returns:
        (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.
    """
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()

    return sum(isinstance(v, bn) for v in model.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model

def info(model, detailed=True, verbose=True, imgsz=224):
    """
    Prints model information.

    Args:
        detailed (bool): if True, prints out detailed information about the model. Defaults to False
        verbose (bool): if True, prints out the model information. Defaults to False
        imgsz (int): the size of the image that the model will be trained on. Defaults to 640
    """
    return model_info(model=model, detailed=detailed, verbose=verbose, imgsz=imgsz)


def fuse(model, verbose=True): # 等价于 thop 计算方式
    """
    Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer, in order to improve the
    computation efficiency.

    Returns:
        (nn.Module): The fused model is returned.
    """
    if not is_fused(thresh=10, model=model):
        for m in model.modules():
            if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, 'bn'):
                if isinstance(m, Conv2):
                    m.fuse_convs()
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
            if isinstance(m, ConvTranspose) and hasattr(m, 'bn'):
                m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
            if isinstance(m, RepConv):
                m.fuse_convs()
                m.forward = m.forward_fuse  # update forward
            if isinstance(m, ConvNormLayer):
                m.conv = fuse_conv_and_bn(m.conv, m.norm)  # update conv
                delattr(m, 'norm')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()
        info(model=model, verbose=verbose)

    return

