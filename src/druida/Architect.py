
import sys  
sys.path.insert(0, '../druida_V01/src/')

import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
#from utils import *

import logging 

from torch.utils.tensorboard import SummaryWriter

from .tools import toolkit
from .tools import utils



logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:

    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cpu"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device) #prepares noise for every step
        self.alpha = 1. - self.beta #calc alpha for every beta step
        self.alpha_hat = torch.cumprod(self.alpha, dim=0) #get alpha hat which is the producof all alphas

    def prepare_noise_schedule(self):
        #this creates a linear distribution of beta along the time steps
        """pending! implement cosine beta distribution"""

        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):

        #X_t sample in any arbitrary time
        #one way to do this is by adding noise step by step 
        #the other way is to doit all at once
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):

        return torch.randint(low=1, high=self.noise_steps, size=(n,))



    """This corresponds to Algorithm Sampling"""
    """We need to sample in reverse direction
    This means from a X_t value, to obtain the previous step.
    And finally obtain X_0"""

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")

        model.eval() #Model used for sampling

        with torch.no_grad():

            """Generate n images with 3 channels."""
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device) #create initial images with random noise

            for i in tqdm(reversed(range(1, self.noise_steps)), position=0): #loop over time steps

                t = (torch.ones(n) * i).long().to(self.device)

                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                #Excluding the first images?!
                if i > 1:
                    noise = torch.randn_like(x) #Returns a tensor with the same size as input that is filled with random numbers from a normal distribution with mean 0 and variance 1.
                else:
                    noise = torch.zeros_like(x)#Returns a tensor with the same size as input that is filled with random numbers from a normal distribution with mean 0 and variance 1.

                #This is the cool formula with epsilon namely the noise perturbation
                #Removing a little noise in each step
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        model.train()
        x = (x.clamp(-1, 1) + 1) / 2 #make all values between 0 and 1 
        x = (x * 255).type(torch.uint8) #valid pixel range
        return x


def train(args):

    utils.setup_logging(args.run_name)

    device = args.device

    dataloader = utils.get_data(args)

    model = toolkit.UNet().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    mse = nn.MSELoss()

    diffusion = Diffusion(img_size=args.image_size, device=device)

    logger = SummaryWriter(os.path.join("runs", args.run_name))

    l = len(dataloader)

    """this corresponds to algorithm 1 """

    for epoch in range(args.epochs):

        logging.info(f"Starting epoch {epoch}:")

        pbar = tqdm(dataloader)

        for i, (images, _) in enumerate(pbar):
            images = images.to(device)

            t = diffusion.sample_timesteps(images.shape[0]).to(device)

            x_t, noise = diffusion.noise_images(images, t)

            predicted_noise = model(x_t, t)

            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            pbar.set_postfix(MSE=loss.item())

            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        sampled_images = diffusion.sample(model, n=images.shape[0])

        utils.save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))



    