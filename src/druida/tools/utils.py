import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import numpy as np
import cv2 as cv
import os
import glob
import ezdxf
import imutils

from ..DataManager import datamanager

    

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
def get_data(image_size,resize, dataset_path,batch_size):

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(resize),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(image_size, scale=(0.9, 0.9)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = torchvision.datasets.ImageFolder(dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    #returns a data loader with normalized images

    return dataloader, dataset.imgs

""""Converting an image to tensor and normalizing"""
def get_data_with_labels(image_size,resize, randomResize, dataset_path,batch_size, drop_last,filter):

    transforms = torchvision.transforms.Compose([
        #torchvision.transforms.Resize(resize),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(image_size, scale=(randomResize, randomResize)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


    cursomDataset = datamanager.CustomDataset(dataset_path,transforms,filter)

    # if not (filter is None):

    #     trainset_1 = torch.utils.data.Subset(cursomDataset, self.filter)
    #     dataloader = DataLoader(trainset_1,  batch_size=batch_size, shuffle=True, drop_last=drop_last)
    # else:
    dataloader = DataLoader(cursomDataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)

    #returns a data loader with normalized images

    return dataloader

def get_data_denormalize(image_size,resize, dataset_path,batch_size):

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(resize),  # args.image_size + 1/4 *args.image_size
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

class CAD():
    def __init__(self, images_folder, destination_folder):
        super().__init__()
        self.images_folder=images_folder
        self.destination_folder=destination_folder
        self.contours_path="./contours/"
        self.output_path=destination_folder

        self.colorContourList=[]

        isExist = os.path.exists(self.contours_path)
        if not isExist:

            # Create a new directory because it does not exist
            os.makedirs(self.contours_path)
            print("The new directory is created!")

        isExist = os.path.exists(self.output_path)
        if not isExist:

            # Create a new directory because it does not exist
            os.makedirs(self.output_path)
            print("The new directory is created!")

    """These functions help to separate 
    contours based on HSV mapping of the image"""

    def colorContour(self,upperBound, lowerBound,image,epsilon_coeff, threshold_Value,contour_name):
        #Channel separation
    
        im = cv.imread(image,cv.IMREAD_UNCHANGED)


        im2 =im.copy()
        im22 = cv.cvtColor(im2,cv.COLOR_BGR2RGB )
        
        hsv = cv.cvtColor(im2, cv.COLOR_BGR2HSV)

        # define range of blue color in HSV
        lower_red = np.array(lowerBound)
        upper_red = np.array(upperBound)
        # Threshold the HSV image to get only blue colors
        mask = cv.inRange(hsv,lower_red,upper_red)

        # Apply threshold on s - use automatic threshold algorithm (use THRESH_OTSU).
        _, thresh = cv.threshold(mask, threshold_Value, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

        # Find contours
        red_cnts = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        red_cnts = imutils.grab_contours(red_cnts) 
        #red_cnts= max(red_cnts, key=cv.contourArea) 
        #Drawing
        canvas = np.zeros_like(im22)
        size = im.shape

        #operating on every contour line
        smoothened=[]

        for contour in red_cnts:
            
            epsilon = epsilon_coeff*cv.arcLength(contour,True)
            cnt_aprox = cv.approxPolyDP(contour,epsilon,True)
        
            smoothened.append(np.asarray(cnt_aprox, dtype=np.int32))
        
            #x,y = contour.T
            # Convert from numpy arrays to normal arrays
            #x = x.tolist()[0]
            #y = y.tolist()[0]
            
            #if len(x)>2:
            
                # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
                #tck, u = splprep([x,y], u=None, s=1.0, per=2,k=1)
                # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
                #u_new = np.linspace(u.min(), u.max(), 250)
                # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
                #x_new, y_new = splev(u_new, tck, der=0)
                # Convert it back to numpy format for opencv to be able to display it
                #res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new,y_new)]
                #smoothened.append(np.asarray(res_array, dtype=np.int32))
            cv.drawContours(canvas, smoothened,-1,(0, 255, 0), 1)

        cv.fillPoly(canvas, pts =smoothened, color=(255,255,255))

        #cv.fillPoly(canvas, pts =red_cnts, color=(255,255,255))

        cv.imwrite(self.contours_path+contour_name+".png", canvas, [cv.IMWRITE_PNG_COMPRESSION, 10]) 
        
        fig, ax = plt.subplots(1, 2, sharex='col', sharey='row',figsize=(5, 5))
        ax[0].imshow(im22)
        ax[1].imshow(canvas)
        
        return red_cnts,size

    def DXF_build(self,multiplier, GoalSize,currentSize,red_cnts, border_cnts,green_cnts, selectedLayer):


        
        dwg = ezdxf.new()#"AC1015"
        msp = dwg.modelspace()

        dwg.layers.new(name="conductor", dxfattribs={"color": 1})
        dwg.layers.new(name="dielectric", dxfattribs={"color": 8})
        dwg.layers.new(name="substrate", dxfattribs={"color": 5})

        """
        Warning! This is the drwaing unit, not the HFSS original units.
        To preserve the resolution everything needs to be rescaled and resized 
        if image size is 512x512 it means 512 nm or 512 mm dpending on the units declared in dxf file
        thus all drawing sizes must be scaled to fit.

        working units together with multipliers must be chosen based on the need to keep ratio and size as we import from HFSS
        """

        scale=multiplier*GoalSize/currentSize

        red_squeezed = [np.squeeze(cnt, axis=1) for cnt in red_cnts]
        inverted_red_squeezed = [scale*arr * [1, -1] for arr in red_squeezed]#*0.1

        green_squeezed = [np.squeeze(cnt, axis=1) for cnt in green_cnts]
        inverted_green_squeezed = [scale*arr * [1, -1] for arr in green_squeezed]#*0.1

        border_squeezed = [np.squeeze(cnt, axis=1) for cnt in border_cnts]
        inverted_border_squeezed = [scale*arr * [1, -1] for arr in border_squeezed]#*0.1

        """Select the layers to export"""
        

        for layer in selectedLayer:
            if layer=="conductor":

                for ctr in inverted_red_squeezed:
                    line=msp.add_lwpolyline(
                        ctr,
                        format="xyb", close=True,
                        dxfattribs={'layer': 'conductor'},
                    )
                    line.dxf.const_width = 0.5
            elif layer=="dielectric":

                for ctr in inverted_green_squeezed:
                    line=msp.add_lwpolyline(
                        ctr,
                        format="xyb", close=True,
                        dxfattribs={'layer': 'dielectric'},
                    )
                    line.dxf.const_width = 0.5
            elif layer=="substrate":

                for ctr in inverted_border_squeezed:
                    line=msp.add_lwpolyline(
                        ctr,
                        format="xyb", close=True,
                        dxfattribs={'layer': 'substrate'},
                    )
                    line.dxf.const_width = 0.5
            else:
                pass

            
            
            """To draw the outline """

        #for ctr in inverted_red_squeezed:
        #    for n in range(len(ctr)):
        #        if n >= len(ctr) - 1:
        #            n = 0
        #            
        #        try:
        #            msp.add_line(ctr[n], ctr[n + 1], dxfattribs={"layer": "red", "lineweight": 30})
        #        except IndexError:
        #            pass



        dwg.saveas(self.output_path+"output.dxf")

    def elevation_file(self,layers,**kwargs):
        
        units=kwargs["units"]
        SIMID=kwargs["simulation_id"]

        file = open(self.output_path+SIMID+'.tech', 'w')
        file.write('units"'+ units+ '"\n')

        for layer in layers:
            
            file.write(layer+" "+ layers[layer]["color"]+" "+ str(layers[layer]["zpos"]) +" "+ str(layers[layer]["thickness"]) +" "+'\n')

        file.close()