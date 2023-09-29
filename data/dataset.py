import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset,DataLoader
from .augmentation import augmentations


class Facades(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.all_files = os.listdir(root_dir)

    def __getitem__(self, item):
        img_file = self.all_files[item]
        image_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(image_path))
        x = image[:, 256:, :]
        y = image[:, :256, :]
        input,target = augmentations.compose(x,y)
        input,target = input.permute(2,0,1),target.permute(2,0,1)
        # print(input.shape,target.shape)
        return input.float(),target.float()

    def __len__(self):
        return len(self.all_files)


batch_size=16

train=Facades("data/facades/facades/train")
validation=Facades("data/facades/facades/train")
train_loader=DataLoader(train,batch_size=batch_size,shuffle=True)
val_loader=DataLoader(validation,batch_size=1,shuffle=False)
