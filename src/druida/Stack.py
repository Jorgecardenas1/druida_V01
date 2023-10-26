import sys  

from .DataManager import datamanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '../druida_V01/src/')

import os
import copy

import torch
import torch.nn as nn

from tqdm import tqdm
from torch import optim
#from utils import *

import logging 

from torch.utils.tensorboard import SummaryWriter

from .tools import toolkit
from .tools import utils

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Predictor:
    pass

class Trainer:
    

    def __init__(self, args):

        self.run_name = args.run_name
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.workers = args.workers
        self.gpu_number=args.gpu_number

        self.checkDevice()
        
    def training(self, trainFunction,testFunction, train_dataloader, test_dataloader, model, loss_fn, optimizer):
        acc=0
        acc_test=0
        loss=0
        test_loss=0
        loss_values = []
        test_loss_values = []
        train_acc_hist = []
        test_acc = []

        for t in range(self.epochs):
            dataiter = iter(train_dataloader)
            testdataiter = iter(test_dataloader)


            print(f"Epoch {t+1}\n-------------------------------")
            acc,loss=trainFunction(next(dataiter), model, loss_fn, optimizer, t,acc,loss)
            acc_test,test_loss=testFunction(next(testdataiter), model, loss_fn,len(train_dataloader), t, acc_test,test_loss)

            loss_values.append(loss)
            test_loss_values.append(test_loss)
            train_acc_hist.append(acc)
            test_acc.append(acc_test)
        print("Done!")

        return loss_values,test_loss_values,train_acc_hist, test_acc

    def checkDevice(self):
        self.device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )


    def multiGPU(self, network):
        print('available Device:'+network.device)
        if (network.device == 'cuda' and (self.gpu_number > 1)):
            network=nn.DataParallel(network,list(range(self.gpu_number)))

        return network
    

    def train_DM(self,args):

        utils.setup_logging(self.run_name)

        device = self.device

        dataloader = utils.get_data(args.image_size, args.dataset_path,self.batch_size)

        model = toolkit.UNet(device=self.device,c_in=3, c_out=3, time_dim=256).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate)

        mse = nn.MSELoss()

        diffusion = Diffusion(img_size=args.image_size, device=device,noise_steps=1000, beta_start=1e-4, beta_end=0.02)

        logger = SummaryWriter(os.path.join("runs", self.run_name))

        l = len(dataloader)

        """this corresponds to algorithm 1 """

        for epoch in range(self.epochs):

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

            utils.save_images(sampled_images, os.path.join("results", self.run_name, f"{epoch}.jpg"))
            torch.save(model.state_dict(), os.path.join("models", self.run_name, f"ckpt.pt"))


    def train_CDM(self,args):

        utils.setup_logging(self.run_name)
        device = self.device
        dataloader = utils.get_data(args.image_size, args.dataset_path,self.batch_size)
        model = toolkit.UNet_conditional(device=self.device,c_in=3, c_out=3, time_dim=256,num_classes=args.num_classes).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate)

        mse = nn.MSELoss()
        diffusion = Diffusion(img_size=args.image_size, device=device)
        logger = SummaryWriter(os.path.join("runs", self.run_name))
        l = len(dataloader)

        ema = toolkit.EMA(0.995)
        ema_model = copy.deepcopy(model).eval().requires_grad_(False)

        for epoch in range(self.epochs):

            logging.info(f"Starting epoch {epoch}:")
            pbar = tqdm(dataloader)

            for i, (images, labels) in enumerate(pbar):
                images = images.to(device)
                labels = labels.to(device)
                t = diffusion.sample_timesteps(images.shape[0]).to(device)
                x_t, noise = diffusion.noise_images(images, t)
                if np.random.random() < 0.1:
                    labels = None
                predicted_noise = model(x_t, t, labels)
                loss = mse(noise, predicted_noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ema.step_ema(ema_model, model)

                pbar.set_postfix(MSE=loss.item())
                logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

            if epoch % 10 == 0:
                labels = torch.arange(2).long().to(device)
                print(labels)
                print(len(labels))
                sampled_images = diffusion.sample_cdm(model, n=len(labels), labels=labels)
                ema_sampled_images = diffusion.sample_cdm(ema_model, n=len(labels), labels=labels)
                utils.plot_images(sampled_images)
                utils.save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
                utils.save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
                torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
                torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
                torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))




    
    
class DNN(nn.Module):

    def __init__(self, layers):
        super(DNN,self).__init__()

        self.checkDevice()
        self.layers = layers
        self.architecture=nn.Sequential()

        for layer in layers:
            self.architecture.add_module(layer['name'],layer['layer'])

        

    def push(self, layer):
        self.layers.append(layer) #each layer must come as dictionary with nn type
        return self.layers
 
    def drop_last(self):
        self.layers.pop() #each layer must come as dictionary with nn type
        return self.layers

    def clear(self):
        self.layers.clear() #each layer must come as dictionary with nn type
        return self.layers
    
    def checkDevice(self):
        self.device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )
    
    def forward(self, input):

        #{"name":"hidden1","layer":nn.Linear(8,120), "type":"hidden", "index",0}
        self.output =  input
        for layer in self.layers:
            action=layer['layer']
            self.output = action(self.output)
                
                
                
        return self.output
    



class Generator(nn.Module):
    def __init__(self, ngpu, input_size, mapping_size, channels ):
        super(Generator, self).__init__()
        

        self.checkDevice()

        self.ngpu = ngpu            

        self.conv1 = nn.ConvTranspose2d(input_size, mapping_size * 8, 6, 1, 0, bias=False)
        self.conv2 = nn.BatchNorm2d(num_features=mapping_size * 8)
        self.conv3 = nn.ReLU(True)
        self.conv4 = nn.ConvTranspose2d(mapping_size * 8, mapping_size * 4, 6, 2, 2, bias=False)
        self.conv5 = nn.BatchNorm2d(mapping_size * 4)
        self.conv6 = nn.ReLU(True)
        self.conv7 = nn.ConvTranspose2d(mapping_size * 4, mapping_size * 2, 6, 2, 4, bias=False)
        self.conv8 = nn.BatchNorm2d(mapping_size * 2)
        self.conv9 = nn.ReLU(True)
        self.conv10 = nn.ConvTranspose2d(mapping_size * 2, mapping_size, 6, 2, 5, bias=False)
        self.conv11 = nn.BatchNorm2d(mapping_size)
        self.conv12 = nn.ReLU(True)
        self.conv13 = nn.ConvTranspose2d(mapping_size, channels, 6, 2, 4, bias=False)
        self.conv14 = nn.Tanh()

    def forward(self, input):
        imageOut = input
        imageOut = self.conv1(imageOut)
        imageOut = self.conv2(imageOut)
        imageOut = self.conv3(imageOut)
        imageOut = self.conv4(imageOut)
        imageOut = self.conv5(imageOut)
        imageOut = self.conv6(imageOut)
        imageOut = self.conv7(imageOut)
        imageOut = self.conv8(imageOut)
        imageOut = self.conv9(imageOut)
        imageOut = self.conv10(imageOut)
        imageOut = self.conv11(imageOut)
        imageOut = self.conv12(imageOut)
        imageOut = self.conv13(imageOut)
        imageOut = self.conv14(imageOut)               
        return imageOut

    def checkDevice(self):
        self.device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )
    


class Discriminator(nn.Module):
    def __init__(self, ngpu=0, image_size=32, discriminator_mapping_size=0, channels=3):
        super(Discriminator, self).__init__()

        
        self.checkDevice()

        self.ngpu = ngpu            
        self.image_size = image_size
        self.channels = channels


        self.l1 = nn.Linear(800, image_size*image_size*channels, bias=False)           
        self.conv1 = nn.Conv2d(2*channels, discriminator_mapping_size, 6, 2, 4, bias=False) 
        self.conv2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(discriminator_mapping_size, discriminator_mapping_size * 2, 6, 2, 5, bias=False)
        self.conv4 = nn.BatchNorm2d(discriminator_mapping_size * 2)
        self.conv5 = nn.LeakyReLU(0.2, inplace=True)
        self.conv6 = nn.Conv2d(discriminator_mapping_size * 2, discriminator_mapping_size * 4, 6, 2, 4, bias=False)
        self.conv7 = nn.BatchNorm2d(discriminator_mapping_size * 4)
        self.conv8 = nn.LeakyReLU(0.2, inplace=True)
        self.conv9 = nn.Conv2d(discriminator_mapping_size * 4, discriminator_mapping_size * 8, 6, 2, 2, bias=False)
        self.conv10 = nn.BatchNorm2d(discriminator_mapping_size * 8)
        self.conv11 = nn.LeakyReLU(0.2, inplace=True)
        self.conv12 = nn.Conv2d(discriminator_mapping_size * 8, 1, 6, 1, 0, bias=False)
        self.conv13 = nn.Sigmoid()


    def forward(self, input, label, b_size):
        x1 = input
        x2 = self.l1(label) #Size must be taken care = 800 in this case
        #the output is imagesize x imagesize x channel
        #hence the need of reshape 

        if self.ngpu == 0 :
        
            x2 = x2.reshape(int(b_size),self.channels,self.image_size,self.image_size) 
        else:
            x2 = x2.reshape(int(b_size/self.ngpu),self.channels,self.image_size,self.image_size) 

        combine = torch.cat((x1,x2),dim=1) # concatenate in a given dimension
        #esto viene del mismo paper sobre los CGAN

        combine = self.conv1(combine) #This conv1 considers 2 x channels from the combine
        combine = self.conv2(combine)
        combine = self.conv3(combine)
        combine = self.conv4(combine)
        combine = self.conv5(combine)
        combine = self.conv6(combine)
        combine = self.conv7(combine)
        combine = self.conv8(combine)
        combine = self.conv9(combine)
        combine = self.conv10(combine)
        combine = self.conv11(combine)
        combine = self.conv12(combine)
        combine = self.conv13(combine)
        return combine

    def checkDevice(self):
        self.device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )
    





class Diffusion:

    def __init__(self,device, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64 ):
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
    

    def sample_cdm(self, model, n, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


