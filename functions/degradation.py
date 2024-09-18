import numpy as np
import torch
from munch import Munch

import functions.svd_operators as svd_op
from utils.inpaint_util import MaskGenerator

__DEGRADATION__ = {}

def register_degradation(name: str):
    def wrapper(fn):
        if __DEGRADATION__.get(name) is not None:
            raise NameError(f'DEGRADATION {name} is already registered')
        __DEGRADATION__[name]=fn
        return fn
    return wrapper

def get_degradation(name: str,
                    deg_config: Munch,
                    device:torch.device,
                    **kwargs):
    if __DEGRADATION__.get(name) is None:
        raise NameError(f'DEGRADATION {name} does not exist.')
    return __DEGRADATION__[name](deg_config, device, **kwargs)

@register_degradation(name='cs_walshhadamard')
def deg_cs_walshhadamard(deg_config, device):
    compressed_size = round(1/deg_config.deg_scale)
    A_funcs = svd_op.WalshHadamardCS(deg_config.channels,
                                     deg_config.image_size,
                                     compressed_size,
                                     torch.randperm(deg_config.image_size**2),
                                     device)
    return A_funcs

@register_degradation(name='cs_blockbased')
def deg_cs_blockbased(deg_config, device):
    cs_ratio = deg_config.deg_scale
    A_funcs = svd_op.CS(deg_config.channels,
                        deg_config.image_size,
                        cs_ratio,
                        device)
    return A_funcs

@register_degradation(name='inpainting')
def deg_inpainting(deg_config, device, mask):
    # TODO: generate mask rather than load
    # loaded = np.load("exp/inp_masks/land1_mask1.npy")  # block
    loaded = mask
    mask = torch.from_numpy(loaded).to(device).reshape(-1)
    missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
    missing_g = missing_r + 1
    missing_b = missing_g + 1
    missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
    A_funcs = svd_op.Inpainting(deg_config.channels,
                                deg_config.image_size,
                                missing,
                                device)
    return A_funcs

@register_degradation(name='denoising')
def deg_denoise(deg_config, device):
    A_funcs = svd_op.Denoising(deg_config.channels,
                               deg_config.image_size,
                               device)
    return A_funcs

@register_degradation(name='colorization')
def deg_colorization(deg_config, device):
    A_funcs = svd_op.Colorization(deg_config.image_size,
                                  device)
    return A_funcs

@register_degradation(name='sr_averagepooling')
def deg_sr_avgpool(deg_config, device):
    blur_by = int(deg_config.deg_scale)
    A_funcs = svd_op.SuperResolution(deg_config.channels,
                                     deg_config.image_size,
                                     blur_by,
                                     device)
    return A_funcs

@register_degradation(name='sr_bicubic')
def deg_sr_bicubic(deg_config, device):
    def bicubic_kernel(x, a=-0.5):
        if abs(x) <= 1:
            return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
        elif 1 < abs(x) and abs(x) < 2:
            return a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
        else:
            return 0

    factor = int(deg_config.deg_scale)
    k = np.zeros((factor * 4))
    for i in range(factor * 4):
        x = (1 / factor) * (i - np.floor(factor * 4 / 2) + 0.5)
        k[i] = bicubic_kernel(x)
    k = k / np.sum(k)
    kernel = torch.from_numpy(k).float().to(device)
    A_funcs = svd_op.SRConv(kernel / kernel.sum(),
                            deg_config.channels,
                            deg_config.image_size,
                            device,
                            stride=factor)
    return A_funcs

@register_degradation(name='deblur_uni')
def deg_deblur_uni(deg_config, device):
    A_funcs = svd_op.Deblurring(torch.tensor([1/36]*36).to(device),
                                deg_config.channels,
                                deg_config.image_size,
                                device)
    return A_funcs

@register_degradation(name='deblur_gauss')
def deg_deblur_gauss(deg_config, device):
    sigma = 5.0
    pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
    # kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(device)
    size = 15
    ker = []
    for k in range(-size, size):
        ker.append(pdf(k))
    kernel = torch.Tensor(ker).to(device)
    A_funcs = svd_op.Deblurring(kernel / kernel.sum(),
                                deg_config.channels,
                                deg_config.image_size,
                                device)
    return A_funcs

@register_degradation(name='deblur_aniso')
def deg_deblur_aniso(deg_config, device):
    sigma = 20
    pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
    kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(device)

    sigma = 1
    pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
    kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(device)

    A_funcs = svd_op.Deblurring2D(kernel1 / kernel1.sum(),
                                  kernel2 / kernel2.sum(),
                                  deg_config.channels,
                                  deg_config.image_size,
                                  device)
    return A_funcs
