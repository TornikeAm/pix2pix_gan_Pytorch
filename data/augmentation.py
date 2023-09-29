import cv2 as cv
import numpy as np
import torch

class Augmentations:
    def __init__(self):
        print("")

    def normalize(self,input_image,target_image):
        inp = (input_image / 255.0)
        tar = (target_image / 255.0)
        return inp,tar


    def random_crop(self,image, dim):
        height, width, _ = dim
        x, y = np.random.uniform(low=0, high=int(height - 256)), np.random.uniform(low=0, high=int(width - 256))
        return image[:, int(x):int(x) + 256, int(y):int(y) + 256]


    def colorJitter(self,input_image,target_image,height = 286,width = 286):

        inp = cv.resize(input_image,(height,width),interpolation=cv.INTER_NEAREST)
        tar = cv.resize(target_image,(height,width),interpolation=cv.INTER_NEAREST)

        #crop
        stack_two_images = np.stack([inp,tar],axis=0)
        # print(stack_two_images.shape)
        crop = self.random_crop(stack_two_images,stack_two_images.shape[1:])
        inp,tar = crop[0],crop[1]

        return  inp,tar

    def compose(self,input_image,target_image):
        image =self.colorJitter(input_image,target_image)
        image = self.normalize(image[0],image[1])
        input,target = torch.tensor(image[0]),torch.tensor(image[1])
        return input,target



augmentations = Augmentations()