"""
This module includes LDM-based inverse problem solvers.
Forward operators follow DPS and DDRM/DDNM.
"""

from typing import Any, Callable, Dict, Optional

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from torchvision import transforms
from tqdm import tqdm

from functions.conjugate_gradient import CG
from functions.svd_operators import A_functions as A_func
from ldm.modules.encoders.modules import FrozenClipImageEmbedder

####### Factory #######
__SOLVER__ = {}

def register_solver(name: str):
    def wrapper(cls):
        if __SOLVER__.get(name, None) is not None:
            raise ValueError(f"Solver {name} already registered.")
        __SOLVER__[name] = cls
        return cls
    return wrapper

def get_solver(name: str, **kwargs):
    if name not in __SOLVER__:
        raise ValueError(f"Solver {name} does not exist.")
    return __SOLVER__[name](**kwargs)

########################


@register_solver("ddim")
class UncondSolver():
    """
    Unconditional solver (i.e. Stble-Diffusion)
    This will generate samples without considering measurements.
    To define LDM functions for solvers.
    """
    def __init__(self,
                 solver_config: Dict,
                 model_key:str="runwayml/stable-diffusion-v1-5",
                 device: Optional[torch.device]=None,
                 **kwargs):
        self.device = device

        # TODO: can we use float16?
        pipe_dtype = torch.float16
        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=pipe_dtype).to(device)
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        total_timesteps = len(self.scheduler.timesteps)
        self.scheduler.set_timesteps(solver_config.num_sampling, device=device)
        self.skip = total_timesteps // solver_config.num_sampling

        self.final_alpha_cumprod = self.scheduler.final_alpha_cumprod.to(device)
        self.scheduler.alphas_cumprod = torch.cat([torch.tensor([1.0]), self.scheduler.alphas_cumprod])

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.sample(*args, **kwargs)

    # DDIM inversion
    @torch.no_grad()
    def inversion(self,
                  z0: torch.Tensor,
                  uc: torch.Tensor,
                  c: torch.Tensor,
                  cfg_guidance: float=1.0,
                  record_all: bool=False):
        """
        DDIM inversion (Hertz et al., 2022, "prompt-to-prompt")

        Args:
            z0 (torch.Tensor): encoded image latent
            uc (torch.Tensor): embedded null text
            c (torch.Tensor): embedded contional text
            cfg_guidance (float): CFG scale
            record_all (bool): if True, return list of latents at all time steps

        Returns:
            (torch.Tensor, None): inversed latent (zT) or list of latents {zt}
            
        """
        # initialize z_0
        zt = z0.clone().to(self.device)

        z_record = [zt.clone()]
         
        # loop
        pbar = tqdm(reversed(self.scheduler.timesteps), desc='DDIM Inversion')
        for step, t in enumerate(pbar):
            prev_t = t - self.skip
            at = self.scheduler.alphas_cumprod[t]
            at_prev = self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else self.final_alpha_cumprod

            noise_pred = self.predict_noise(zt, t, uc, c, cfg_guidance) 
            z0t = (zt - (1-at_prev).sqrt() * noise_pred) / at_prev.sqrt()
            zt = at.sqrt() * z0t + (1-at).sqrt() * noise_pred

            if record_all:
                z_record.append(zt)
        
        if record_all:
            zt = z_record

        return zt
    
    def initialize_latent(self,
                          inversion: bool=False,
                          src_img: Optional[torch.Tensor]=None,
                          **kwargs):
        if inversion:
            z = self.inversion(self.encode(src_img),
                               kwargs.get('uc'),
                               kwargs.get('c'),
                               cfg_guidance=kwargs.get('cfg_guidance', 1.0),
                               record_all=kwargs.get('record_all', False))
        else:
            z = torch.randn((1, 4, 64, 64)).to(self.device)
        return z.requires_grad_()

    def predict_noise(self,
                      zt: torch.Tensor,
                      t: torch.Tensor,
                      uc: torch.Tensor,
                      c: torch.Tensor,
                      cfg_guidance: float):
        """
        compuate epsilon_theta with CFG.
        args:
            zt (torch.Tensor): latent features
            t (torch.Tensor): timestep
            uc (torch.Tensor): null-text embedding
            c (torch.Tensor): text embedding
            cfg_guidance (float): CFG value
        """
        if uc is None or cfg_guidance == 1.0:
            t_in = t.unsqueeze(0)
            noise_pred = self.unet(zt, t_in, encoder_hidden_states=c)['sample']
        elif c is None or cfg_guidance == 0.0:
            t_in = t.unsqueeze(0)
            noise_pred = self.unet(zt, t_in, encoder_hidden_states=uc)['sample']
        else:
            c_embed = torch.cat([uc, c], dim=0)
            z_in = torch.cat([zt] * 2) 
            t_in = torch.cat([t.unsqueeze(0)] * 2)
            noise_pred = self.unet(z_in, t_in, encoder_hidden_states=c_embed)['sample']
            noise_uc, noise_c = noise_pred.chunk(2)
            noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
        return noise_pred

    @torch.no_grad()
    def get_text_embed(self, null_prompt, prompt):
        """
        Get text embedding.
        args:
            null_prompt (str): null text
            prompt (str): guidance text
        """
        # null text embedding (negation)
        null_text_input = self.tokenizer(null_prompt,
                                         padding='max_length',
                                         max_length=self.tokenizer.model_max_length,
                                         return_tensors="pt",)
        null_text_embed = self.text_encoder(null_text_input.input_ids.to(self.device))[0]

        # text embedding (guidance)
        text_input = self.tokenizer(prompt,
                                    padding='max_length',
                                    max_length=self.tokenizer.model_max_length,
                                    return_tensors="pt",
                                    truncation=True)
        text_embed = self.text_encoder(text_input.input_ids.to(self.device))[0]

        return null_text_embed, text_embed

    def encode(self, x):
        """
        xt -> zt
        """
        return self.vae.encode(x).latent_dist.sample() * 0.18215

    def decode(self, zt):
        """
        zt -> xt
        """
        zt = 1/0.18215 * zt
        img = self.vae.decode(zt).sample.float()
        return img

    @torch.autocast(device_type='cuda', dtype=torch.float16) 
    def sample(self,
               cfg_guidance=7.5,
               prompt=["",""],
               **kwargs):
        """
        Main function that defines each solver.
        This will generate samples without considering measurements.
        """
        
        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])

        # Initialize zT
        latent_dim = kwargs.get('latent_dim', 64)
        zt = torch.randn((1, 4, latent_dim, latent_dim)).to(self.device)
        zt = zt.requires_grad_()

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="SD")
        for _, t in enumerate(pbar):
            prev_t = t - self.skip
            at = self.scheduler.alphas_cumprod[t]
            at_prev = self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else self.final_alpha_cumprod

            with torch.no_grad():
                noise_pred = self.predict_noise(zt, t, uc, c, cfg_guidance)
            
            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_pred
        
        # for the last step, do not add noise
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()

@register_solver(name='max_guide')
class MaxGuideSolver(UncondSolver):
    def predict_noise(self,
                      zt: torch.Tensor,
                      t: torch.Tensor,
                      uc: torch.Tensor,
                      c: torch.Tensor,
                      cfg_guidance: float):
        """
        compuate epsilon_theta with CFG.
        args:
            zt (torch.Tensor): latent features
            t (torch.Tensor): timestep
            uc (torch.Tensor): null-text embedding
            c (torch.Tensor): text embedding
            cfg_guidance (float): CFG value
        """
        c_embed = torch.cat([uc, c], dim=0)
        z_in = torch.cat([zt] * 2) 
        t_in = torch.cat([t.unsqueeze(0)] * 2)
        noise_pred = self.unet(z_in, t_in, encoder_hidden_states=c_embed)['sample']
        noise_uc, noise_c = noise_pred.chunk(2)
        return noise_uc, noise_c

    def optim_latent(self, zt, t, uc, c, cfg_guidance):
        zt = zt.detach().requires_grad_()
        optim = torch.optim.Adam([zt], lr=5e-4)
        for _ in range(5):
            optim.zero_grad()
            noise_uc, noise_c = self.predict_noise(zt, t, uc, c, cfg_guidance)
            # loss = -((noise_c - noise_uc.detach())**2).mean()
            loss = -(torch.linalg.norm((noise_c-noise_uc.detach()).reshape(-1)).mean())
            # loss = torch.linalg.norm((noise_uc-noise_c.detach()).reshape(-1)).mean()
            # loss = -(noise_c - noise_uc.detach()).mean()
            loss.backward()
            optim.step()
        return zt


    @torch.autocast(device_type='cuda', dtype=torch.float16) 
    def sample(self,
               cfg_guidance=7.5,
               prompt=["",""],
               **kwargs):
        """
        Main function that defines each solver.
        This will generate samples without considering measurements.
        """
        
        show = kwargs.get('show', False)

        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])

        # Initialize zT
        latent_dim = kwargs.get('latent_dim', 64)
        zt = torch.randn((1, 4, latent_dim, latent_dim)).to(self.device)
        zt = zt.requires_grad_()

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="SD")
        for step, t in enumerate(pbar):
            prev_t = t - self.skip
            at = self.scheduler.alphas_cumprod[t]
            at_prev = self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else self.final_alpha_cumprod

            with torch.no_grad():
                noise_uc, _ = self.predict_noise(zt, t, uc, c, cfg_guidance)

            if step < len(self.scheduler.timesteps)/3:
                zt = self.optim_latent(zt, t, uc, c, cfg_guidance)

            with torch.no_grad():
                noise_uc2, noise_c2 = self.predict_noise(zt, t, uc, c, cfg_guidance)
                noise_pred = noise_uc + cfg_guidance * (noise_c2 - noise_uc2)

                loss = torch.nn.functional.mse_loss(noise_c2, noise_uc2).mean().item()
                pbar.set_postfix({'loss': loss})
            
            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_uc

            if show and (step+1) % 10 == 0:
                with torch.no_grad():
                    x0t = self.decode(z0t)
                clear_output(wait=True)
                img = (x0t / 2.0 + 0.5).clamp(0, 1).detach().cpu()
                img = (img.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
                img = Image.fromarray(img[0])
                display(img)
        
        # for the last step, do not add noise
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()


@register_solver(name='treg')
class TRegSolver(UncondSolver):
    def __init__(self,
                 solver_config: Dict,
                 model_key:str="runwayml/stable-diffusion-v1-5",
                 device: Optional[torch.device]=None,
                 **kwargs):
        super().__init__(solver_config, model_key, device, **kwargs)
        
        self.clip_img_enc = FrozenClipImageEmbedder(model='ViT-L/14',
                                                    device=device)
    
    @torch.no_grad()
    def data_consistency(self,
                         z0t: torch.Tensor,
                         measurement: torch.Tensor,
                         A: Callable,
                         At: Callable,
                         cg_lamb: float=1e-4):
        """
        Apply data consistency update via conjugate gradient.
        args:
            z0t (torch.Tensor): denoised estimate
            measurement (torch.Tensor): measurement
            A (Callable): forward operator
            At (Callable): adjoint operator
        """
        x0t = self.decode(z0t).detach()
        bvec = At(measurement) + cg_lamb * x0t.reshape(1, -1)
        x0y = CG(A=lambda x: At(A(x)) + cg_lamb * x,
                      b=bvec,
                      x=x0t,
                      m=10)
        z0y = self.encode(x0y)
        return z0y, x0y
           
    def adaptive_negation(self,
                          x0: torch.Tensor,
                          uc: torch.Tensor,
                          lr: float=1e-3,
                          num_iter: int=10):
        """
        Update null-text embedding to minimize the similarity in CLIP space.
        args:
            x0 (torch.Tensor): input image
            uc (torch.Tensor): null-text embedding
            lr (float): learning rate
            num_iter (int): number of iterations
        """
        uc = uc.detach()
        x0 = x0.detach()
        img_feats = self.clip_img_enc(x0).detach()
        img_feats = img_feats / img_feats.norm(dim=1, keepdim=True) # normalize

        uc.requires_grad = True
        optim = torch.optim.Adam([uc], lr=lr)

        for _ in range(num_iter):
            optim.zero_grad()
            sim = img_feats @ uc.permute(0,2,1)
            loss = sim.mean()
            loss.backward(retain_graph=True)
            optim.step()

        return uc

    @torch.autocast(device_type='cuda', dtype=torch.float16) 
    def sample(self,
               measurement: torch.Tensor,
               operator: A_func,
               cfg_guidance: float=7.5,
               prompt: list[str]=["", ""],
               **kwargs):
        """
        Solve inverse problem with TReg.
        """

        use_DPS = kwargs.get('use_DPS', False)
        cg_lamb = kwargs.get('cg_lamb', 1e-4)

        A = operator.A
        At = operator.At

        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
        
        # Initialize zT
        # if record_all = True, zt is a list of latents at all time steps
        # in that case, zt[-1] is the final latent (zT)
        x_src = At(measurement).reshape(1, 3, 512, 512)
        zt = self.initialize_latent(inversion=False,
                                    src_img=x_src,
                                    uc=uc,
                                    c=c,
                                    record_all=False)

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="TReg")
        for step, t in enumerate(pbar):
            prev_t = t - self.skip
            at = self.scheduler.alphas_cumprod[t]
            at_prev = self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else self.final_alpha_cumprod

            if step % 3 == 0 and step < 170:
                with torch.no_grad():
                    noise_pred = self.predict_noise(zt, t, uc, c, cfg_guidance)
                z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

                # Data consistency update
                z0y, x0y = self.data_consistency(z0t, measurement, A, At, cg_lamb)

                with torch.no_grad():
                    if (step+1) % 10 == 0:
                        clear_output(wait=True)
                        img = (x0y / 2.0 + 0.5).clamp(0, 1).detach().cpu()
                        img = (img.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
                        img = Image.fromarray(img[0])
                        display(img)

                # adaptive negation
                uc = self.adaptive_negation(x0y, uc)
                
                # DDPM-like noise adding
                noise = torch.randn_like(z0y).to(self.device)
                z0_ema = at_prev * z0y + (1-at_prev) * z0t
                zt = at_prev.sqrt() * z0_ema + (1-at_prev) * noise_pred
                zt = zt + (1-at_prev).sqrt() * at_prev.sqrt() * noise
            
            else:
                if use_DPS:
                    dps_lamb = at_prev.sqrt()

                    noise_pred = self.predict_noise(zt, t, uc, c, cfg_guidance=0)
                    z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()
                    zt_prime = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_pred

                    x0t = self.decode(z0t)
                    residue = torch.linalg.norm((measurement - A(x0t)).reshape(-1))
                    grad = torch.autograd.grad(residue, zt)[0]
                    zt = zt_prime - dps_lamb * grad
                else:
                    with torch.no_grad():
                        noise_pred = self.predict_noise(zt, t, uc, c, cfg_guidance)
                    z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()
                    zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_pred
        
        # for the last step, do not add noise
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()

import numpy as np
from IPython.display import clear_output, display
from PIL import Image


@register_solver(name='control')
class ControlSolver(TRegSolver):
    def predict_noise(self,
                      zt: torch.Tensor,
                      t: torch.Tensor,
                      uc: torch.Tensor,
                      c: torch.Tensor,
                      cfg_guidance: float):
        """
        compuate epsilon_theta with CFG.
        args:
            zt (torch.Tensor): latent features
            t (torch.Tensor): timestep
            uc (torch.Tensor): null-text embedding
            c (torch.Tensor): text embedding
            cfg_guidance (float): CFG value
        """
        if uc is None or cfg_guidance == 1.0:
            t_in = t.unsqueeze(0)
            noise_pred = self.unet(zt, t_in, encoder_hidden_states=c)['sample']
            return noise_pred
        elif c is None or cfg_guidance == 0.0:
            t_in = t.unsqueeze(0)
            noise_pred = self.unet(zt, t_in, encoder_hidden_states=uc)['sample']
            return noise_pred
        else:
            c_embed = torch.cat([uc, c], dim=0)
            z_in = torch.cat([zt] * 2) 
            t_in = torch.cat([t.unsqueeze(0)] * 2)
            noise_pred = self.unet(z_in, t_in, encoder_hidden_states=c_embed)['sample']
            noise_uc, noise_c = noise_pred.chunk(2)
        return noise_uc, noise_c
            
    def clip_preprocess(self, img: torch.Tensor) -> torch.Tensor:
        # resize and crop
        tf = transforms.Compose([
            transforms.Resize(224, antialias=True),
            transforms.CenterCrop(224),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        img = tf(img)

        return img

    def clip_sim(self, img, text_embed):
        img = img / 2.0 + 0.5  # (0, 1)
        img = self.clip_preprocess(img)
        img_embed = self.clip_img_enc(img).float()

        img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
        similarity = torch.nn.CosineSimilarity(dim=-1)(img_embed, text_embed)
        # similarity = text_embed @ img_embed.T  # shape = (1, 77, 1)
        return similarity[0, :]

    def clip_img_sim(self, img1, img2):
        img1 = img1 / 2.0 + 0.5  # (0, 1)
        img1 = self.clip_preprocess(img1)
        img1_embed = self.clip_img_enc(img1).float()

        img2 = img2 / 2.0 + 0.5  # (0, 1)
        img2 = self.clip_preprocess(img2)
        img2_embed = self.clip_img_enc(img2).float()

        img1_embed = img1_embed / img1_embed.norm(dim=-1, keepdim=True)
        img1_embed = img2_embed / img2_embed.norm(dim=-1, keepdim=True)
        similarity = torch.nn.CosineSimilarity(dim=-1)(img1_embed, img2_embed)
        # similarity = text_embed @ img_embed.T  # shape = (1, 77, 1)
        return similarity

    def data_consistency(self, z0t, y, A):
        z0t = z0t.detach().requires_grad_()
        optim = torch.optim.Adam([z0t], lr=1e-4)
        for _ in range(5):
            optim.zero_grad()
            x0t = self.decode(z0t)
            loss = torch.linalg.norm((A(x0t) - y).reshape(-1)).mean()
            # loss = torch.nn.functional.mse_loss(A(x0t), y)
            loss.backward(retain_graph=True)
            optim.step()
        return z0t, loss, z0t.grad

    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self,
               src_img: torch.Tensor,
               cond_operator: torch.nn.Module,
               cfg_guidance: float=7.5,
               prompt: list[str]=["", ""],
               **kwargs):
        """
        Image editing via DDS.
        prompt[0] is null text, prompt[1] is source text, prompt[2] is target text.
        """
        show = kwargs.get('display', False)
        A = lambda x: cond_operator((x/2+0.5)) # assume input is in [-1, 1]

        # count number of tokens in prompt
        num_token = len(prompt[1].split())
        
        # Text embedding
        uc, c_tgt = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
        
        # Initialize zT
        # if record_all = True, zt is a list of latents at all time steps
        # in that case, zt[-1] is the final latent (zT)
        zt = self.initialize_latent(inversion=kwargs.get('inversion', False),
                                    src_img=src_img,
                                    uc=uc,
                                    c=c_tgt,
                                    cfg_guidance=0.0,
                                    record_all=False)
        
        condition = A(src_img).detach()

        # Sampling
        noise = torch.randn_like(zt).to(self.device)
        loss_list = []
        grad_norm = []
        pbar = tqdm(self.scheduler.timesteps, desc="DreamControl")
        for step, t in enumerate(pbar):
            prev_t = t - self.skip
            at = self.scheduler.alphas_cumprod[t]
            at_prev = self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else self.final_alpha_cumprod

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, c_tgt, cfg_guidance=cfg_guidance)
                noise_cfg = noise_uc + cfg_guidance * (noise_c - noise_uc)

            z0t = (zt - (1-at).sqrt() * noise_cfg) / at.sqrt()
            z0t = z0t.detach().requires_grad_()
            for _ in range(3):
                x0t = self.decode(z0t)
                loss = torch.linalg.norm((A(x0t) - condition).reshape(-1)).mean()
                dist_loss = (noise - noise_cfg) * z0t
                loss = at * loss + (1-at) * dist_loss.mean()
                grad = torch.autograd.grad(loss, z0t)[0]
                z0t = z0t - at.sqrt() * grad

            noise = (1-at_prev) * noise_cfg + (1-at_prev).sqrt() * at_prev.sqrt() * torch.randn_like(z0t).to(self.device)
            noise = noise / (1-at_prev).sqrt()
            zt = at_prev.sqrt() * z0t + (1-at_prev) * noise
            
            #zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_cfg

            loss = loss.item()
            grad = torch.linalg.norm(grad).item()

            pbar.set_postfix({'loss': loss, 'grad': grad})
            loss_list.append(loss)
            grad_norm.append(grad)

            if show and (step+1) % 10 == 0:
                with torch.no_grad():
                    x0t = self.decode(z0t)
                clear_output(wait=True)
                img = (x0t / 2.0 + 0.5).clamp(0, 1).detach().cpu()
                img = (img.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
                img = Image.fromarray(img[0])
                display(img)

        # for the last step, do not add noise
        with torch.no_grad():
            img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        
        # output
        output = {
            'image': img.detach().cpu(),
            'loss': loss_list,
            'grad_norm': grad_norm
        }

        return output


@register_solver(name='dds_optim')
class DDSOptimSolver(TRegSolver):
    def __init__(self,
                 solver_config: Dict,
                 model_key:str="runwayml/stable-diffusion-v1-5",
                 device='cuda',
                 **kwargs):
        super().__init__(solver_config, model_key, device, **kwargs)
        self.alpha_exp = 0
        self.sigma_exp = 0
        self.t_min = 50
        self.t_max = 950

    def noise_input(self, z, eps=None, timestep: Optional[int] = None):
        if timestep is None:
            b = z.shape[0]
            timestep = torch.randint(
                low=self.t_min,
                high=min(self.t_max, 1000) - 1,  # Avoid the highest timestep.
                size=(b,),
                dtype=torch.long)
        if eps is None:
            eps = torch.randn_like(z).to(z.device)
        at = self.scheduler.alphas_cumprod[timestep.detach().cpu()].to(z.device)
        z_t = at.sqrt() * z + (1-at).sqrt() * eps
        return z_t, eps, timestep.to(z.device), at
    
    def dds_loss(self,
                 z_src,
                 z_tgt,
                 uc,
                 c_src,
                 c_tgt,
                 cfg_guidance,
                 t=None,
                 eps=None):


        with torch.inference_mode():
            zt_src, eps, timestep, at  = self.noise_input(z_src, eps, t)
            zt_tgt, _, _, _ = self.noise_input(z_tgt, eps, timestep)

            z_in = torch.cat([zt_src, zt_tgt] * 2)
            t_in = torch.cat([timestep] * 4)
            c_embed = torch.cat([uc, uc, c_src, c_tgt], dim=0)
            noise_pred = self.unet(z_in, t_in, encoder_hidden_states=c_embed)['sample']
            noise_uc, noise_c = noise_pred.chunk(2)
            noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            eps_src, eps_tgt = noise_pred.chunk(2)

            grad = (at.sqrt() ** self.alpha_exp) * ((1-at).sqrt() ** self.sigma_exp) * (eps_tgt - eps_src)

        loss = z_tgt * grad.detach().clone()
        loss = loss.sum() / (z_tgt.shape[2] * z_tgt.shape[3])
        return loss

    def latent_update(self, z_src, z_tgt, uc, c_src, c_tgt, at, t, lamb1, lamb2, cfg_guidance=7.5):
        z_tgt = z_tgt.detach().requires_grad_()
        z0t = z_tgt.clone()
        z_optim = torch.optim.SGD([z_tgt], lr=1e-1)

        for _ in range(3):
            loss = self.dds_loss(z_src,
                                 z_tgt,
                                 uc,
                                 c_src,
                                 c_tgt,
                                 cfg_guidance=cfg_guidance,
                                 t=t.unsqueeze(0))
            loss = lamb1 * loss * (1-at).sqrt() + lamb2 * (z_tgt - z0t.detach()).pow(2).mean() * at.sqrt()

            z_optim.zero_grad()
            loss.backward()
            z_optim.step()

        return z_tgt, self.decode(z_tgt)

    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self,
               src_img: torch.Tensor,
               cfg_guidance: float=7.5,
               prompt: list[str]=["", "", ""],
               latent_optim_kwargs: dict={},
               **kwargs):
        """
        Image editing via DDS.
        prompt[0] is null text, prompt[1] is source text, prompt[2] is target text.
        """
        update = kwargs.get('update_range', lambda x: True)
        display = kwargs.get('display', True)
        
        # Text embedding
        uc, c_src = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
        _, c_tgt = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[2])
        
        # initialize z_src
        with torch.no_grad():
            z_src = self.encode(src_img)
        
        # Initialize zT
        # if record_all = True, zt is a list of latents at all time steps
        # in that case, zt[-1] is the final latent (zT)
        zt = self.initialize_latent(inversion=kwargs.get('inversion', True),
                                    src_img=src_img,
                                    uc=uc,
                                    c=c_src,
                                    cfg_guidance=0.0,
                                    record_all=False)

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="DDS")
        for step, t in enumerate(pbar):
            prev_t = t - self.skip
            at = self.scheduler.alphas_cumprod[t]
            at_prev = self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else self.final_alpha_cumprod

            if update(step):
                with torch.no_grad():
                    noise_pred = self.predict_noise(zt, t, uc, None, cfg_guidance=0.0)
                z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

                z0t, x0t = self.latent_update(z_src=z_src,
                                              z_tgt=z0t,
                                              uc=uc,
                                              c_src=c_src,
                                              c_tgt=c_tgt,
                                              at=at,
                                              t=t,
                                              cfg_guidance=cfg_guidance,
                                              **latent_optim_kwargs)

                if display and (step+1) % 10 == 0:
                    clear_output(wait=True)
                    img = (x0t / 2.0 + 0.5).clamp(0, 1).detach().cpu()
                    img = (img.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
                    img = Image.fromarray(img[0])
                    display(img)

                # adaptive negation
                if kwargs.get('update_null', False):
                    c_src = self.adaptive_negation(x0t, c_src, lr=kwargs.get('null_lr', 1e-3))

                zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_pred
            
            else:
                with torch.no_grad():
                    noise_pred = self.predict_noise(zt, t, uc, c_tgt, cfg_guidance=0.0)
                z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()
                zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_pred

                if display and (step+1) % 10 == 0:
                    clear_output(wait=True)
                    with torch.no_grad():
                        x0t = self.decode(z0t)
                    img = (x0t / 2.0 + 0.5).clamp(0, 1).detach().cpu()
                    img = (img.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
                    img = Image.fromarray(img[0])
                    display(img)
        
        # for the last step, do not add noise
        with torch.no_grad():
            img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()


@register_solver(name='dream_edit')
class DreamSamplerEdit(DDSOptimSolver):
    def predict_noise(self,
                      zt: torch.Tensor,
                      t: torch.Tensor,
                      uc: torch.Tensor,
                      c: torch.Tensor,
                      cfg_guidance: float):
        """
        compuate epsilon_theta with CFG.
        args:
            zt (torch.Tensor): latent features
            t (torch.Tensor): timestep
            uc (torch.Tensor): null-text embedding
            c (torch.Tensor): text embedding
            cfg_guidance (float): CFG value
        """
        if uc is None or cfg_guidance == 1.0:
            t_in = t.unsqueeze(0)
            noise_pred = self.unet(zt, t_in, encoder_hidden_states=c)['sample']
            return noise_pred
        elif c is None or cfg_guidance == 0.0:
            t_in = t.unsqueeze(0)
            noise_pred = self.unet(zt, t_in, encoder_hidden_states=uc)['sample']
            return noise_pred
        else:
            c_embed = torch.cat([uc, c], dim=0)
            z_in = torch.cat([zt] * 2) 
            t_in = torch.cat([t.unsqueeze(0)] * 2)
            noise_pred = self.unet(z_in, t_in, encoder_hidden_states=c_embed)['sample']
            noise_uc, noise_c = noise_pred.chunk(2)
            noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
        return noise_uc, noise_pred
            

    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self,
               src_img: torch.Tensor,
               cfg_guidance: float=7.5,
               prompt: list[str]=["", "", ""],
               **kwargs):
        """
        Image editing via DDS.
        prompt[0] is null text, prompt[1] is source text, prompt[2] is target text.
        """
        update = kwargs.get('update_range', lambda x: True)
        show = kwargs.get('display', True)
        
        # Text embedding
        uc, c_src = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
        _, c_tgt = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[2])
        
        # Initialize zT
        # if record_all = True, zt is a list of latents at all time steps
        # in that case, zt[-1] is the final latent (zT)
        zt = self.initialize_latent(inversion=kwargs.get('inversion', True),
                                    src_img=src_img,
                                    uc=uc,
                                    c=c_src,
                                    cfg_guidance=0.0,
                                    record_all=False)

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="DreamSampler")
        for step, t in enumerate(pbar):
            prev_t = t - self.skip
            at = self.scheduler.alphas_cumprod[t]
            at_prev = self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else self.final_alpha_cumprod

            if update(step):
                with torch.no_grad():
                    noise_uc, noise_cfg = self.predict_noise(zt, t, uc, c_tgt, cfg_guidance=at * cfg_guidance)
                
                z0t = (zt - (1-at).sqrt() * noise_cfg) / at.sqrt()

                with torch.no_grad():
                    x0t = self.decode(z0t)

                if show and (step+1) % 10 == 0:
                    clear_output(wait=True)
                    img = (x0t / 2.0 + 0.5).clamp(0, 1).detach().cpu()
                    img = (img.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
                    img = Image.fromarray(img[0])
                    display(img)

                zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_uc
            
            else:
                with torch.no_grad():
                    noise_pred = self.predict_noise(zt, t, uc, c_tgt, cfg_guidance=0.0)
                z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()
                zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_pred

                if show and (step+1) % 10 == 0:
                    clear_output(wait=True)
                    with torch.no_grad():
                        x0t = self.decode(z0t)
                    img = (x0t / 2.0 + 0.5).clamp(0, 1).detach().cpu()
                    img = (img.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
                    img = Image.fromarray(img[0])
                    display(img)
        
        # for the last step, do not add noise
        with torch.no_grad():
            img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()


@register_solver(name='psld')
class PSLDSolver(TRegSolver):
    @torch.autocast(device_type='cuda', dtype=torch.float16) 
    def sample(self,
               measurement: torch.Tensor,
               operator: A_func,
               cfg_guidance: float=7.5,
               prompt: list[str]=["", ""],
               **kwargs):
        """
        Solve inverse problem with TReg.
        """

        show = kwargs.get('display', True)

        A = operator.A
        At = operator.At

        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
        
        # Initialize zT
        # if record_all = True, zt is a list of latents at all time steps
        # in that case, zt[-1] is the final latent (zT)
        x_src = At(measurement).reshape(1, 3, 512, 512)
        zt = self.initialize_latent(inversion=False,
                                    src_img=x_src,
                                    uc=uc,
                                    c=c,
                                    record_all=False)

        orig_shape = x_src.shape

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="PSLD")
        for step, t in enumerate(pbar):
            prev_t = t - self.skip
            at = self.scheduler.alphas_cumprod[t]
            at_prev = self.scheduler.alphas_cumprod[prev_t] if prev_t > 0 else self.final_alpha_cumprod

            with torch.no_grad():
                noise_pred = self.predict_noise(zt, t, uc, c, cfg_guidance)
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()
            zt_prev = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_pred

            # data consistency
            x_pred = self.decode(z0t)
            y_pred = A(x_pred)
            y_residue = torch.linalg.norm((y_pred - measurement).reshape(1, -1))
            
            ortho_project = x_pred.reshape(1, -1) - At(y_pred)
            parallel_project = At(measurement)
            projected = parallel_project + ortho_project

            recon_z = self.encode(projected.reshape(orig_shape))
            z0_residue = torch.linalg.norm((recon_z - z0t).reshape(1, -1))
            
            omega = 1.0
            gamma = 0.5

            residue = omega * y_residue + gamma * z0_residue
            grad = torch.autograd.grad(residue, zt)[0]
            zt = zt_prev - grad

            with torch.no_grad():
                if (step+1) % 10 == 0 and show:
                    clear_output(wait=True)
                    img = (x_pred.reshape(orig_shape) / 2.0 + 0.5).clamp(0, 1).detach().cpu()
                    img = (img.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
                    img = Image.fromarray(img[0])
                    display(img)
        
        # for the last step, do not add noise
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()

@register_solver(name='dream_inverse')
class DistilationMultipleInverseSolver(TRegSolver):
    def predict_noise(self,
                      zt: torch.Tensor,
                      t: torch.Tensor,
                      uc: torch.Tensor,
                      c: torch.Tensor,
                      cfg_guidance: float):
        """
        compuate epsilon_theta with CFG.
        args:
            zt (torch.Tensor): latent features
            t (torch.Tensor): timestep
            uc (torch.Tensor): null-text embedding
            c (torch.Tensor): text embedding
            cfg_guidance (float): CFG value
        """
        if uc is None or cfg_guidance == 1.0:
            t_in = t.unsqueeze(0)
            noise_pred = self.unet(zt, t_in, encoder_hidden_states=c)['sample']
            return noise_pred
        elif c is None or cfg_guidance == 0.0:
            t_in = t.unsqueeze(0)
            noise_pred = self.unet(zt, t_in, encoder_hidden_states=uc)['sample']
            return noise_pred
        else:
            c_embed = torch.cat([uc, c], dim=0)
            z_in = torch.cat([zt] * 2) 
            t_in = torch.cat([t] * 2)
            noise_pred = self.unet(z_in, t_in, encoder_hidden_states=c_embed)['sample']
            noise_uc, noise_c = noise_pred.chunk(2)
            noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            return noise_uc, noise_c

    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self,
               measurement: torch.Tensor,
               operator: A_func,
               cfg_guidance: float=7.5,
               prompt: list[str]=["", ["", ""]],
               **kwargs):
        """
        Image editing via DDS.
        prompt[0] is null text, prompt[1] is source text, prompt[2] is target text.
        """
        update = kwargs.get('update_range', lambda x: True)
        show = kwargs.get('display', True)
        use_DPS = kwargs.get('use_DPS', False)
        cg_lamb = kwargs.get('cg_lamb', 1e-4)
        
        A = operator.A
        At = operator.At
        
        # Text embedding
        null_prompt = prompt[0]
        tgt_prompts = prompt[1]
        mask_list = kwargs.get('mask')
        mask = torch.stack(mask_list, dim=0).unsqueeze(1)

        assert len(tgt_prompts) == len(mask_list), "each mask requires a target prompt"

        uc, _ = self.get_text_embed(null_prompt=null_prompt, prompt=null_prompt)
        c_tgt_list = [self.get_text_embed(null_prompt=null_prompt, prompt=c_tgt)[1] for c_tgt in tgt_prompts]
        c_tgt = torch.cat(c_tgt_list, dim=0)

        # Initialize zT
        # if record_all = True, zt is a list of latents at all time steps
        # in that case, zt[-1] is the final latent (zT)
        y = At(measurement).reshape(1, 3, 512, 512)
        zt = self.initialize_latent(inversion=kwargs.get('inversion', True),
                                    src_img=y,
                                    uc=uc,
                                    c=uc,
                                    cfg_guidance=0.0,
                                    record_all=False)

        mask_prod = mask.prod(dim=0, keepdim=True)
        zt = mask_prod * zt + (1-mask_prod) * torch.randn_like(zt).to(self.device)

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="Multiple-inpaint")
        for step, t in enumerate(pbar):
            prev_t = t - self.skip
            at = self.scheduler.alphas_cumprod[t]
            at_prev = self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else self.final_alpha_cumprod

            if update(step):
                with torch.no_grad():
                    noise_uc_t, noise_c_t = self.predict_noise(
                        zt.tile(len(tgt_prompts), 1, 1, 1),  # len, 64, 64,
                        t.unsqueeze(0).tile(len(tgt_prompts)),
                        uc.tile(len(tgt_prompts), 1, 1),
                        c_tgt,
                        cfg_guidance=cfg_guidance
                        )
                    # noise_cfg_t = noise_uc_t + cfg_guidance * (noise_c_t - noise_uc_t)
                    noise_uc = noise_uc_t.mean(dim=0, keepdim=True)

                z0t_uc = (zt - (1-at).sqrt() * noise_uc) / at.sqrt()
                z0t_c = (zt - (1-at).sqrt() * noise_c_t) / at.sqrt()

                z0y, _ = self.data_consistency(z0t_uc, measurement, A, At, cg_lamb)
                z0t = at * z0y + (1-at) * z0t_uc

                # multiply each mask
                z0t = mask_prod * z0t + (1-mask_prod) * ((1-at) * z0t) + ((1-mask) * (at * z0t_c)).sum(dim=0, keepdim=True)

                with torch.no_grad():
                    x0t = self.decode(z0t)

                if show and (step+1) % 10 == 0:
                    clear_output(wait=True)
                    img = (x0t / 2.0 + 0.5).clamp(0, 1).detach().cpu()
                    img = (img.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
                    img = Image.fromarray(img[0])
                    display(img)

                # adaptive negation
                if kwargs.get('update_null', False):
                    uc = self.adaptive_negation(x0t, uc.mean(0, keepdim=True), lr=kwargs.get('null_lr', 1e-3))

                noise = (1-at).sqrt() * noise_uc + at.sqrt() * torch.randn_like(z0t).to(self.device)
                zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise

                # noise = (1-at).sqrt() * noise_uc_t + at.sqrt() * torch.randn_like(z0t).to(self.device)
                # zt = at.sqrt() * z0t + (1-at).sqrt() * noise
            
            else:
                if use_DPS:
                    dps_lamb = at_prev.sqrt()
                    noise_uc_t = self.predict_noise(zt, t, uc, uc, cfg_guidance=0.0)
                    z0t = (zt - (1-at).sqrt() * noise_uc_t) / at.sqrt()
                    zt_prime = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_uc_t

                    x0t = self.decode(z0t)
                    residue = torch.linalg.norm((measurement - A(x0t)).reshape(-1))
                    grad = torch.autograd.grad(residue, zt)[0]
                    zt = zt_prime - dps_lamb * grad

                else:
                    with torch.no_grad():
                        noise_pred = self.predict_noise(zt, t, uc, uc, cfg_guidance=0.0)
                    z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()
                    zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_pred

                if show and (step+1) % 10 == 0:
                    clear_output(wait=True)
                    with torch.no_grad():
                        x0t = self.decode(z0t)
                    img = (x0t / 2.0 + 0.5).clamp(0, 1).detach().cpu()
                    img = (img.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
                    img = Image.fromarray(img[0])
                    display(img)
        
        # for the last step, do not add noise
        with torch.no_grad():
            img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()
