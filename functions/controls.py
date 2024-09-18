import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image
from transformers import pipeline

from utils.control_util import get_thin_kernels

### FACTORY FUNCTIONS
__CONTOLS__ = {}

def register_controls(name: str):
    def wrapper(cls):
        if __CONTOLS__.get(name) is not None:
            raise ValueError(f"Control {name} already exists")
        __CONTOLS__[name] = cls
        return cls
    return wrapper

def get_control(name: str):
    if __CONTOLS__.get(name) is None:
        raise ValueError(f"Control {name} not found")
    return __CONTOLS__[name]

######################

@register_controls('canny_edge')
class CannyEdge(nn.Module):
    def __init__(self, device, low_threshold=None, high_threshold=None):
        super().__init__()
        self.device = device
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

        # for step 1
        self.gaussian_filter = transforms.GaussianBlur(kernel_size=5, sigma=(1.4, 1.4))

        # for step 2
        sobel_filter = torch.tensor([[1, 0, -1],
                                     [2, 0, -2],
                                     [1, 0, -1]], dtype=torch.float32).to(self.device)
        self.sobel_filter = sobel_filter.unsqueeze(0).unsqueeze(0).repeat((3,3,1,1)) # assume RGB input

        # for step 3
        thin_kernels = get_thin_kernels()
        directional_kernel = torch.stack([torch.tensor(kernel) for kernel in thin_kernels], dim=0)
        self.directional_kernel = directional_kernel.unsqueeze(1).repeat((1, 3, 1, 1)).float().to(self.device)

        # for step 4
        hyteresis_kernel = torch.ones((3, 3)).to(self.device) + 0.25
        self.hyteresis_kernel = hyteresis_kernel.unsqueeze(0).unsqueeze(0).repeat((3, 3, 1, 1))

    def forward(self, img:torch.Tensor, hysteresis=False):
        img = img.to(self.device)

        # 1. apply Gaussian filter
        img = self.gaussian_filter(img)

        # 2. gradient compute
        G_x = F.conv2d(img, self.sobel_filter, padding=1)
        G_y = F.conv2d(img, self.sobel_filter.mT, padding=1)
        mag = torch.sqrt(G_x**2 + G_y**2)
        theta = torch.atan2(G_y, G_x)

        # 3. non-maximum suppression
        theta = theta * 180 / np.pi + 180  # convert to degree (0-360)
        theta = theta // 45
        directional = nn.functional.conv2d(mag, self.directional_kernel, padding=1)

        pos_idx = (theta % 8).long()
        thin_edges = mag.clone()
        for ni in range(4):
            pi = ni + 4
            is_oriented_i = (pi == pos_idx)
            is_oriented_i = is_oriented_i + (pos_idx == ni)

            # takes pi-th channel value from directional using gather
            pi_tensor = torch.tensor(pi).repeat((directional.shape[0], 1, directional.shape[2], directional.shape[3])).to(self.device)
            ni_tensor = torch.tensor(ni).repeat((directional.shape[0], 1, directional.shape[2], directional.shape[3])).to(self.device)
            pos_direction = torch.gather(directional, 1, pi_tensor)
            neg_direction = torch.gather(directional, 1, ni_tensor)
            selected_direction = torch.cat([pos_direction, neg_direction], dim=0)

            # get local maximum pixels for the angle
            is_max = selected_direction.min(dim=0)[0] > 0.0
            is_max = is_max.unsqueeze(1)

            # apply non maxumum suppression
            mask = (is_max == 0) * (is_oriented_i > 0)
            mask = ~mask
            thin_edges = thin_edges * mask

        # 4. thresholding
        if self.low_threshold is not None:
            low = F.relu(torch.sign(thin_edges - self.low_threshold))
        
            if self.high_threshold is not None:
                high = F.relu(torch.sign(thin_edges - self.high_threshold))
            
                thin_edges = low * 0.5 + high * 0.5

                if hysteresis:
                    # get weaks and check if they are high or not
                    weak = (thin_edges == 0.5) * 1
                    weak_is_high = (F.conv2d(thin_edges, self.hyteresis_kernel, padding=1) > 1) * weak
                    thin_edges = high + weak_is_high

            thin_edges = low
        
        return thin_edges

@register_controls('sobel_edge')
class SobelEdge(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        sobel_filter = torch.tensor([[1, 0, -1],
                                     [2, 0, -2],
                                     [1, 0, -1]], dtype=torch.float32).to(self.device)
        self.sobel_filter = sobel_filter.unsqueeze(0).unsqueeze(0).repeat((3,3,1,1))

    def forward(self, img:torch.Tensor):
        img = img.to(self.device)
        G_x = F.conv2d(img, self.sobel_filter, padding=1)
        G_y = F.conv2d(img, self.sobel_filter.mT, padding=1)
        mag = torch.sqrt(G_x**2 + G_y**2)

        # normalize
        # mag = mag / mag.max()
        mag = (mag - mag.min()) / (mag.max() - mag.min())
        return mag

@register_controls('depth_map')
class DepthMap(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        model_type = "DPT_Large"
        self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        self.model = self.model.to(device)
        self.model.eval()
        
        midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type == 'DPT_Large' or 'DPT_Hybrid':
            self.transform = midas_transform.dpt_transform
        else:
            self.transform = midas_transform.small_transform

    def forward(self, img: torch.Tensor):
        #img = self.transform(img).to(self.device)
        img = img.to(self.device)
        predict = self.model(img)
        predict = F.interpolate(predict.unsqueeze(1), size=img.shape[-2:], mode="bicubic", align_corners=False)
        return predict

if __name__ == '__main__':
    img = np.array(Image.open('samples/control/married.jpeg').convert('RGB'))
    img = torch.from_numpy(img).permute(2, 0, 1)/255.0
    img = img.requires_grad_(True).unsqueeze(0)
    
    control = get_control('sobel_edge')('cuda')
    sobel_edge = control(img).cpu()#.repeat(1, 3, 1, 1)

    control = get_control('canny_edge')('cuda', low_threshold=150/255, high_threshold=200/255)
    canny_edge = control(img).cpu()#.repeat(1, 3, 1, 1)

    save_image(torch.cat([img, sobel_edge, canny_edge]), "control_test.png", normalize=True, scale_each=True)

