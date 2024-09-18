'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
from functools import partial

from torch.nn import functional as F
from torchvision import torch

from utils.blur_util import Blurkernel
from utils.img_util import fft2d
from utils.resizer import Resizer

# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def noisy_forward(self, data, **kwargs):
        # calculate A * X + n
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass

    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


@register_operator(name='noise')
class DenoiseOperator(LinearOperator):
    def __init__(self, device):
        self.device = device
    
    def forward(self, data):
        return data

    def noisy_forward(self, data):
        return data

    def transpose(self, data):
        return data
    
    def ortho_project(self, data):
        return data

    def project(self, data):
        return data


@register_operator(name='sr_bicubic')
class SuperResolutionOperator(LinearOperator):
    def __init__(self,
                 in_shape, 
                 scale_factor,
                 noise,
                 noise_scale,
                 device):
        self.device = device
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)
        self.down_sample = Resizer(in_shape, 1/scale_factor).to(device)
        self.noise = get_noise(name=noise, scale=noise_scale)

    def A(self, data, **kwargs):
        return self.forward(data, **kwargs)

    def forward(self, data, **kwargs):
        return self.down_sample(data)
    
    def noisy_forward(self, data, **kwargs):
        return self.noise.forward(self.down_sample(data))

    def transpose(self, data, **kwargs):
        return self.up_sample(data)

    def project(self, data, measurement, **kwargs):
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)

@register_operator(name='deblur_motion')
class MotionBlurOperator(LinearOperator):
    def __init__(self,
                 kernel_size,
                 intensity,
                 noise,
                 noise_scale,
                 device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='motion',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)
        self.noise = get_noise(name=noise, scale=noise_scale)
    
    def forward(self, data, **kwargs):
        # A^T * A 
        return self.conv(data)

    def noisy_forward(self, data, **kwargs):
        return self.noise.forward(self.conv(data))

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        kernel = self.conv.get_kernel().type(torch.float32).to(self.device)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)

    def apply_kernel(self, data, kernel):
        self.conv.update_weights(kernel.type(torch.float32))
        return self.conv(data)

@register_operator(name='deblur_gauss')
class GaussialBlurOperator(LinearOperator):
    def __init__(self,
                 kernel_size,
                 intensity,
                 noise,
                 noise_scale,
                 device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))
        self.noise = get_noise(name=noise, scale=noise_scale)

    def forward(self, data, **kwargs):
        return self.conv(data)
    
    def noisy_forward(self, data, **kwargs):
        return self.noise.forward(self.conv(data))

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)
    
    def apply_kernel(self, data, kernel):
        self.conv.update_weights(kernel.type(torch.float32))
        return self.conv(data)

@register_operator(name='inpainting')
class InpaintingOperator(LinearOperator):
    '''This operator get pre-defined mask and return masked image.'''
    def __init__(self,
                 noise,
                 noise_scale,
                 device):
        self.device = device
        self.noise = get_noise(name=noise, scale=noise_scale)
    
    def forward(self, data, **kwargs):
        try:
            return data * kwargs.get('mask', None).to(self.device)
        except:
            raise ValueError("Require mask")

    def noisy_forward(self, data, **kwargs):
        return self.noise.forward(self.forward(data, **kwargs))

    def transpose(self, data, **kwargs):
        return data
    
    def ortho_project(self, data, **kwargs):
        return data - self.forward(data, **kwargs)

# Operator for BlindDPS.
@register_operator(name='blind_blur')
class BlindBlurOperator(LinearOperator):
    def __init__(self, device, **kwargs) -> None:
        self.device = device
    
    def forward(self, data, kernel, **kwargs):
        return self.apply_kernel(data, kernel)

    def transpose(self, data, **kwargs):
        return data
    
    def apply_kernel(self, data, kernel):
        #TODO: faster way to apply conv?:W
        
        b_img = torch.zeros_like(data).to(self.device)
        for i in range(3):
            b_img[:, i, :, :] = F.conv2d(data[:, i:i+1, :, :], kernel, padding='same')
        return b_img


class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    @abstractmethod
    def noisy_forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 

@register_operator(name='phase_retrieval')
class PhaseRetrievalOperator(NonLinearOperator):
    def __init__(self,
                 oversample,
                 noise,
                 noise_scale,
                 device):
        self.pad = int((oversample / 8.0) * 256)
        self.device = device
        self.noise = get_noise(name=noise, scale=noise_scale)
        
    def forward(self, data, **kwargs):
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        amplitude = fft2d(padded).abs()
        return amplitude

    def noisy_forard(self, data, **kwargs):
        return self.noise.forward(self.forward(data, **kwargs))

# =============
# Noise classes
# =============


__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='clean')
class Clean(Noise):
    def __init__(self, **kwargs):
        pass

    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, scale):
        self.scale = scale
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.scale


@register_noise(name='poisson')
class PoissonNoise(Noise):
    def __init__(self, scale):
        self.scale = scale

    def forward(self, data):
        '''
        Follow skimage.util.random_noise.
        '''

        # version 3 (stack-overflow)
        import numpy as np
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.scale) / 255.0 / self.scale)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)
