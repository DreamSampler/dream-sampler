"""
This module includes LDM-based inverse problem solvers.
Forward operators follow DPS and DDRM/DDNM.
"""

from typing import Any, Callable, Dict, Optional

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
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

class StableDiffusion():
    def __init__(self,
                 solver_config: Dict,
                 model_key:str="botp/stable-diffusion-v1-5",
                 device: Optional[torch.device]=None,
                 **kwargs):
        self.device = device

        self.dtype = kwargs.get("pipe_dtype", torch.float16)
        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.dtype).to(device)
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
        self.sample(*args, **kwargs)

    def sample(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Solver must implement sample() method.")
    
    def alpha(self, t):
        at = self.scheduler.alphas_cumprod[t] if t >= 0 else self.final_alpha_cumprod
        return at

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

    def predict_noise(self,
                      zt: torch.Tensor,
                      t: torch.Tensor,
                      uc: torch.Tensor,
                      c: torch.Tensor):
        """
        compuate epsilon_theta for null and condition
        args:
            zt (torch.Tensor): latent features
            t (torch.Tensor): timestep
            uc (torch.Tensor): null-text embedding
            c (torch.Tensor): text embedding
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if uc is None:
            noise_c = self.unet(zt, t, encoder_hidden_states=c)['sample']
            noise_uc = noise_c
        elif c is None:
            noise_uc = self.unet(zt, t, encoder_hidden_states=uc)['sample']
            noise_c = noise_uc
        else:
            c_embed = torch.cat([uc, c], dim=0)
            z_in = torch.cat([zt] * 2) 
            t_in = torch.cat([t] * 2)
            noise_pred = self.unet(z_in, t_in, encoder_hidden_states=c_embed)['sample']
            noise_uc, noise_c = noise_pred.chunk(2)

        return noise_uc, noise_c

    @torch.no_grad()
    def inversion(self,
                  z0: torch.Tensor,
                  uc: torch.Tensor,
                  c: torch.Tensor,
                  cfg_guidance: float=1.0):

        # initialize z_0
        zt = z0.clone().to(self.device)
         
        # loop
        pbar = tqdm(reversed(self.scheduler.timesteps), desc='DDIM Inversion')
        for _, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            noise_uc, noise_c = self.predict_noise(zt, t, uc, c) 
            noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            z0t = (zt - (1-at_prev).sqrt() * noise_pred) / at_prev.sqrt()
            zt = at.sqrt() * z0t + (1-at).sqrt() * noise_pred

        return zt
    
    def initialize_latent(self,
                          method: str='random',
                          src_img: Optional[torch.Tensor]=None,
                          **kwargs):
        if method == 'ddim':
            z = self.inversion(self.encode(src_img.to(self.dtype).to(self.device)),
                               kwargs.get('uc'),
                               kwargs.get('c'),
                               cfg_guidance=kwargs.get('cfg_guidance', 0.0))
        elif method == 'npi':
            z = self.inversion(self.encode(src_img.to(self.dtype).to(self.device)),
                               kwargs.get('c'),
                               kwargs.get('c'),
                               cfg_guidance=1.0)
        elif method == 'random':
            size = kwargs.get('latent_dim', (1, 4, 64, 64))
            z = torch.randn(size).to(self.device)
        else: 
            raise NotImplementedError

        return z.requires_grad_()

###########################################
# Base version
###########################################

@register_solver("ddim")
class BaseDDIM(StableDiffusion):
    """
    Basic DDIM solver for SD.
    Useful for text-to-image generation
    """

    @torch.autocast(device_type='cuda', dtype=torch.float16) 
    def sample(self,
               cfg_guidance=7.5,
               prompt=["",""],
               callback_fn=None,
               **kwargs):
        """
        Main function that defines each solver.
        This will generate samples without considering measurements.
        """
        
        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])

        # Initialize zT
        zt = self.initialize_latent()
        zt = zt.requires_grad_()

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="SD")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            
            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_pred

            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]
        
        # for the last step, do not add noise
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()

@register_solver("ddim_inversion")
class InversionDDIM(BaseDDIM):
    """
    Editing via WardSwap after inversion.
    Useful for text-guided image editing.
    """
    @torch.autocast(device_type='cuda', dtype=torch.float16) 
    def sample(self,
               src_img,
               cfg_guidance=7.5,
               prompt=["","",""],
               callback_fn=None,
               **kwargs):
        
        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])

        # Initialize zT
        zt = self.initialize_latent(method='ddim',
                                    src_img=src_img,
                                    uc=uc,
                                    c=c,
                                    cfg_guidance=cfg_guidance)
        zt = zt.requires_grad_()

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="SD")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
            
            # tweedie
            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()

            # add noise
            zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_pred

            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]
        
        # for the last step, do not add noise
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()

@register_solver(name='dream_edit')
class DreamSamplerEdit(BaseDDIM):
    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self,
               src_img: torch.Tensor,
               cfg_guidance: float=7.5,
               prompt: list[str]=["", "", ""],
               callback_fn=None,
               **kwargs):
        """
        Image editing via DDS.
        prompt[0] is null text, prompt[1] is source text, prompt[2] is target text.
        """
        
        # Text embedding
        uc, src_c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
        _, tgt_c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[2])
        
        # Initialize zT
        zt = self.initialize_latent(method='ddim', src_img=src_img,
                                    uc=uc, c=src_c, cfg_guidance=0.0)

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="DreamSampler-Edit")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, tgt_c)
                noise_pred = noise_uc + at * cfg_guidance * (noise_c - noise_uc)

            z0t = (zt - (1-at).sqrt() * noise_pred) / at.sqrt()
            zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_uc
        
            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]
        # for the last step, do not add noise
        with torch.no_grad():
            img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()

@register_solver(name='dream_inpaint')
class DreamSamplerInpaint(BaseDDIM):
    @torch.no_grad()
    def data_consistency(self,
                         z0t: torch.Tensor,
                         measurement: torch.Tensor,
                         A: Callable,
                         At: Callable,
                         cg_lamb: float=1e-4):
        x0t = self.decode(z0t).detach()
        bvec = At(measurement) + cg_lamb * x0t.reshape(1, -1)
        x0y = CG(A=lambda x: At(A(x)) + cg_lamb * x,
                      b=bvec,
                      x=x0t,
                      m=10)
        z0y = self.encode(x0y)
        return z0y, x0y

    @torch.autocast(device_type='cuda', dtype=torch.float16)
    def sample(self,
               measurement: torch.Tensor,
               operator: A_func,
               cfg_guidance: float=7.5,
               prompt: list[str]=["", "",[""]],
               masks: list[torch.Tensor]=[],
               callback_fn=None,
               **kwargs):

        no_dps = kwargs.get('no_dps', lambda step: True)

        A = operator.A
        At = operator.At
        mask = torch.stack(masks, dim=0).unsqueeze(1)  # (N, 1, H, W)
        num_masks = mask.shape[0]

        # Text embedding
        tgt_texts = prompt[2]
        assert len(tgt_texts) == num_masks, "each mask requires a target text"

        uc, src_c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
        c_tgt_list = [self.get_text_embed(null_prompt=prompt[0], prompt=tgt)[1] for tgt in tgt_texts]
        tgt_c = torch.cat(c_tgt_list, dim=0)
        
        # Initialize zT
        zt = self.initialize_latent(method='ddim',
                                    src_img=At(measurement).reshape(1, 3, 512, 512),
                                    uc=uc,
                                    c=src_c,
                                    cfg_guidance=0.0)
        
        mask_prod = mask.prod(dim=0, keepdim=True)  # (1, 1, H, W)

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="DreamSampler-Inpaint")
        for step, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            if no_dps(step):
                with torch.no_grad():
                    noise_uc, noise_c = self.predict_noise(
                        zt.tile(num_masks, 1, 1, 1),
                        t.unsqueeze(0).tile(num_masks),
                        uc.tile(num_masks, 1, 1),
                        tgt_c,
                        )
                    noise_uc = noise_uc.mean(dim=0, keepdim=True)

                z0t_uc = (zt - (1-at).sqrt() * noise_uc) / at.sqrt() # (1, C, H, W)
                z0t_c = (zt- (1-at).sqrt() * noise_c) / at.sqrt() # (num_masks, C, H, W)

                # data consistency
                z0y, _ = self.data_consistency(z0t_uc, measurement, A, At, cg_lamb=1e-4)
                z0t = at * z0y + (1-at) * z0t_uc

                # distillation (text-guidance)
                z0t = mask_prod * z0t + (1-mask_prod) * ((1-at) * z0t) + ((1-mask) * (at * z0t_c)).sum(dim=0, keepdim=True)

                noise = (1-at).sqrt() * noise_uc + at.sqrt() * torch.randn_like(z0t).to(self.device)
                zt = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise
            else:
                dps_lamb = at_prev.sqrt()

                noise_uc, _ = self.predict_noise(zt, t, uc, uc)
                z0t = (zt - (1-at).sqrt() * noise_uc) / at.sqrt()
                zt_prime = at_prev.sqrt() * z0t + (1-at_prev).sqrt() * noise_uc

                x0t = self.decode(z0t)
                residue = torch.linalg.norm((measurement - A(x0t)).reshape(-1))
                grad = torch.autograd.grad(residue, zt)[0]
                zt = zt_prime - dps_lamb * grad

            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]
    
        # for the last step, do not add noise
        with torch.no_grad():
            img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()

###########################################
# Other algorithms
###########################################

@register_solver(name='psld')
class PSLDSolver(BaseDDIM):
    @torch.autocast(device_type='cuda', dtype=torch.float16) 
    def sample(self,
               measurement: torch.Tensor,
               operator: A_func,
               cfg_guidance: float=7.5,
               prompt: list[str]=["", ""],
               callback_fn=None,
               **kwargs):
        """
        Solve inverse problem with PSLD.
        """

        A = operator.A
        At = operator.At
        omega = kwargs.get('omega', 1.0)
        gamma = kwargs.get('gamma', 0.5)

        with torch.no_grad():
            x_src = self.decode(zt)
            orig_shape = x_src.shape

        # Text embedding
        uc, c = self.get_text_embed(null_prompt=prompt[0], prompt=prompt[1])
        
        # Initialize zT
        zt = self.initialize_latent()

        # Sampling
        pbar = tqdm(self.scheduler.timesteps, desc="PSLD")
        for step, t in enumerate(pbar):
            prev_t = t - self.skip
            at = self.scheduler.alphas_cumprod[t]
            at_prev = self.scheduler.alphas_cumprod[prev_t] if prev_t > 0 else self.final_alpha_cumprod

            noise_uc, noise_c = self.predict_noise(zt, t, uc, c)
            noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)
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
            
            residue = omega * y_residue + gamma * z0_residue
            grad = torch.autograd.grad(residue, zt)[0]
            zt = zt_prev - grad

            if callback_fn is not None:
                callback_kwargs = {'z0t': z0t.detach(),
                                    'zt': zt.detach(),
                                    'decode': self.decode}
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        # for the last step, do not add noise
        img = self.decode(z0t)
        img = (img / 2 + 0.5).clamp(0, 1)
        return img.detach().cpu()



#############################

if __name__ == "__main__":
    # print all list of solvers
    print(f"Possble solvers: {[x for x in __SOLVER__.keys()]}")
    
