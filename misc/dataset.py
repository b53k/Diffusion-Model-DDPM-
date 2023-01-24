'''Dataset and DataLoader'''

import torch, os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class Dataset(Dataset):
    def __init__(self, path, img_size: int = 32, lim: int = 10):
        # img_size = image size
        # lim = first n images to be loaded in CPU/GPU to avoid intensive resource usage
        self.sizes = [img_size, img_size]
        items, labels = [], []

        for data in os.listdir(path)[:lim]:
            # path = './Linnaeus_Flower'
            # data = 500_32.jpg (for example)
            item = os.path.join(path, data)
            items.append(item)
            labels.append(data)
        
        self.items = items
        self.labels = labels
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        data = Image.open(self.items[index].convert('RGB'))
        data = np.asarray(transforms.Resize(self.sizes)(data))      # [32,32,3]
        data = 2*(data/255.0) - 1                                   # scales between -1 and 1
        data = torch.from_numpy(data).permute(2,0,1)                # reshape to tensor [3,32,32]
        return data, self.labels[index]

'''
ds = Dataset('./Linnaeus_Flower/')
dataloader = DataLoader(ds, 10, True, drop_last = False)

for i, j in dataloader:
    print ('curr bs: ', len(i))
    print (j)
    new_img = i.permute(0,2,3,1)

Need to change back to 0,1 integer type for display

import matplotlib.pyplot as plt
new_img = new_img.type(torch.uint8)

plt.figure(figsize = (10,2))
for i in range(len(new_img)):
    plt.subplot(1,10,i+1)
    plt.imshow(new_img[i])
'''