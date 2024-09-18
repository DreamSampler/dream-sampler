import argparse

import numpy as np
import torch
from munch import munchify
from PIL import Image
from torchvision.utils import save_image

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
    parser.add_argument('--img_path', type=str, default='samples/horse_1.jpg')
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--null_prompt', type=str, default="")
    parser.add_argument('--prompt', type=str, default="zebra")
    parser.add_argument('--cfg_guidance', type=float, default=0.2)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--NFE', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # control randomness 
    set_seed(args.seed)

    # set device
    device = 'cpu' if args.cpu else 'cuda'

    # prepare solver
    solver = get_solver(name='dream_edit',
                        device=device,
                        solver_config=munchify({'num_sampling': args.NFE}))

    # load image
    img = load_img(args.img_path, args.img_size)
    img = torch.from_numpy(img).permute(2, 0, 1) / 127.5 - 1
    img = img.unsqueeze(0).to(device)

    # solve inverse problem
    recon = solver.sample(src_img = img,
                   prompt=[args.null_prompt,
                           args.null_prompt,
                           args.prompt],
                   cfg_guidance=args.cfg_guidance,
                   update_null=False,
                   display=False,
                   inversion=True)

    # save result using torchvision.utils.save_image
    save_image(recon, 'edited.png', normalize=True)

if __name__ == '__main__':
    main()
