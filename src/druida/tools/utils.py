import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import numpy as np
import cv2
import os
import glob

def plot_images(images):

    
    plt.figure(figsize=(16, 16))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)



""""Converting an image to tensor and normalizing"""
def get_data(image_size, dataset_path,batch_size):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = torchvision.datasets.ImageFolder(dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #returns a data loader with normalized images

    return dataloader

def get_data_denormalize(image_size, dataset_path,batch_size):

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),

    ])

    #Ojo realmente si se necesita normalizar transforms.Normalize((0.1307,), (0.3081,))
    dataset = torchvision.datasets.ImageFolder(dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #for inputs, targets in dataloader:
    #    print(inputs.size())
    #    print(targets.size())

    #returns a data loader with normalized images

    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

def load_images(path):
    loadedImages = []
    # return array of images
    filenames = glob.glob(path)
    filenames.sort()
    for imgdata in filenames:
        # determine whether it is an image.
        if os.path.isfile(os.path.splitext(os.path.join(path, imgdata))[0] + ".png"):
            img_array = cv2.imread(os.path.join(path, imgdata))
            img_array = np.float32(img_array)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            loadedImages.append(img_array)
    return loadedImages

class Binary:
    
    def convert(self,results_folder):
        # set folder of images
        folder = results_folder  # Set 
        path = folder+'*-bw.png'
        filenames = glob.glob(path)
        filenames.sort()
        imgs = load_images(path) # load images
        imgs = np.asarray(imgs)
        print(np.shape(imgs))
        
        for i in range(len(filenames)):
            basename = os.path.basename(filenames[i])
            # Turn image array into binary array (black to 1, white to 0)
            binary = np.zeros(shape = (np.shape(imgs)[1], np.shape(imgs)[2]), dtype = np.uint8)
            img = imgs[i][:][:]
            print(np.shape(binary))
            print(np.shape(img[:][:][0]))
            binary[img[:][:] <= 50] = 1
            
            print(len(binary[binary==1]))
            # doubling the amount of pixels in both dimensions
            resize_fac = 2
            height, width = imgs.shape[:][:][1:]
            new = np.ones(shape = (resize_fac*height, resize_fac*width), dtype = np.uint8)
            print(np.shape(new))
            print(np.shape(binary))
            new[:binary.shape[0],:binary.shape[1]] = binary[:][:]
            for i in range(height-1,-1,-1):
                for j in range(width-1,-1,-1):
                    cur = new[i][j]
                    new[resize_fac*i:resize_fac*(i+1),resize_fac*j:resize_fac*(j+1)] = [[cur]*resize_fac] * resize_fac
                        
            print(len(new[new==1]))
            print(len(new[new==1])/(resize_fac**2) == len(binary[binary==1]))
            
            print(folder+basename[:-4]+'.txt')
            file1 = open(folder+basename[:-4]+'.txt', 'w')
            header = str(height*resize_fac)+' 1 '+str(height*resize_fac)+' \n'+str(width*resize_fac)+' 1 '+str(width*resize_fac)+' \n2 1 2\n'
            file1.write(header)
            body = new.reshape(new.shape[0]*new.shape[1],1)
            cnt=0
            for item in body:
                file1.write("%s\n" % item[0])
                cnt +=1
            for item in body:
                file1.write("%s\n" % item[0])
            
            file1.close()