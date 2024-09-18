import argparse

import numpy as np
import torch
from munch import munchify
from PIL import Image
from torchvision.utils import save_image

from functions.degradation import get_degradation
from solver.latent_diffusion import get_solver


def set_seed(seed: int):
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def load_img(img_path: str, size: int=512):
    image = np.array(Image.open(img_path).convert('RGB').resize((size, size)))
    return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='samples/ffhq_1.png')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--null_prompt', type=str, default="out of focus, depth of field")
    parser.add_argument('--prompt', type=str, default="glasses")
    parser.add_argument('--cfg_guidance', type=float, default=5.0)
    parser.add_argument('--op_type', type=str, default='inpainting')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--NFE', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # control randomness 
    set_seed(args.seed)

    # set device
    device = 'cpu' if args.cpu else 'cuda'

    # prepare solver
    solver = get_solver(name='dream_inpaint',
                        device=device,
                        solver_config=munchify({'num_sampling': args.NFE}))

    # prepare operator config
    op_config = munchify({
        'image_size': args.img_size,
        'deg_scale': 16,
        'channels': 3
    })

    mask = np.load('inp_masks/mask_eye.npy')
    mask[mask>0] = 1

    # perpare operator
    operator = get_degradation(
        name=args.op_type,
        device=device,
        deg_config=op_config,
        mask=mask
    )

    # load image
    img = load_img(args.img_path, args.img_size)
    img = torch.from_numpy(img).permute(2, 0, 1) / 127.5 - 1
    img = img.unsqueeze(0).to(device)

    # measurement
    y = operator.A(img)
    y = y + 0.01 * torch.randn_like(y)
    save_image(operator.At(y).reshape(1,3,512,512), 'input.png', normalize=True)

    # solve inverse problem
    recon = solver.sample(measurement=y,
                   operator=operator,
                   prompt=[args.null_prompt,
                           args.null_prompt,
                           [args.prompt]],
                   cfg_guidance=args.cfg_guidance,
                   inversion=True,
                   update_null=False,
                   cg_lamb=5e-4,
                   use_DPS=True,
                   masks=[torch.from_numpy(mask[::8, ::8]).float().to('cuda')],
                   no_dps=lambda x: x%3 ==0 and x < 170,
                   display=False)

    # save result using torchvision.utils.save_image
    save_image(recon, 'recon.png', normalize=True)

if __name__ == '__main__':
    main()
