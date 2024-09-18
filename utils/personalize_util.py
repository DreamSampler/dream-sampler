from typing import Optional

import torch
from diffusers import DDIMScheduler


class TextualInverter:
    def __init__(self,
                 latent: torch.Tensor,
                 scheduler: DDIMScheduler,
                 device: str='cuda',
                 lr : float=1e-4,
                 adam_weight_decay: float=1e-2,
                 adam_epsilon: float=1e-8,
                 embed: Optional[torch.Tensor]=None,
                 embed_size: int=768):

        self.device = device
        self.latent = latent.to(device)
        self.scheduler = scheduler
        self.num_timesteps = scheduler.timesteps[-1]
        self.init_embed(embed, embed_size)
        self.optim = torch.optim.AdamW([self.embed],
                                       lr=lr,
                                       weight_decay=adam_weight_decay,
                                       eps=adam_epsilon)

    def init_embed(self, embed: Optional[torch.Tensor]=None, embed_size: Optional[int]=768):
        if embed is None:
            self.embed = torch.randn(embed_size, requires_grad=True).to(self.device)
        else:
            self.embed = embed.clone().to(self.device)

        self.embed = self.embed.unsqueeze(0).unsqueeze(0)
        self.embed = self.embed.detach()
        self.embed.requires_grad_(True)

    @property 
    def get_embed(self):
        return self.embed.clone().detach()
    
    def step(self):
        self.optim.step()
        self.optim.zero_grad()

    def backward(self, unet: torch.nn.Module, timestep:Optional[torch.Tensor]=None, noise:Optional[torch.Tensor]=None):
        bs = self.latent.shape[0]

        if timestep is None:
            idx = torch.randint(0, len(self.scheduler.timesteps), size=(bs,))
            timestep = self.scheduler.timesteps[idx].cpu()
            # timestep = torch.randint(0, self.num_timesteps, size=(bs,))

        if noise is None:
            noise = torch.randn_like(self.latent).to(self.device)
        else:
            noise = noise.to(self.device)

        unet.requires_grad_(False)

        at = self.scheduler.alphas_cumprod[timestep].to(self.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        zt = at.sqrt() * self.latent + (1 - at).sqrt() * noise
        pred = unet(zt, timestep.to(self.device), encoder_hidden_states=self.embed.tile(bs,1,1))['sample']

        loss = torch.nn.functional.mse_loss(pred, noise)
        loss.backward()

        return loss.item()

    def save_embed(self, path: str):
        try:
            torch.save(self.embed.detach().cpu(), path)
        except Exception as e:
            print(f'Failed to save embed, due to {e}.')

    
