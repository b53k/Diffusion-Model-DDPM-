'''Dataset and DataLoader'''

import torch, os
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
import numpy as np

class Dataset(Dataset):
    def __init__(self, path, img_size: int = 32):
        # img_size = image size
        # lim = first n images to be loaded in CPU/GPU to avoid intensive resource usage
        self.sizes = [img_size, img_size]
        items, labels = [], []

        for data in os.listdir(path):
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
        data = Image.open(self.items[index]).convert('RGB')
        data = np.asarray(transforms.Resize(self.sizes)(data))      # [32,32,3]
        data = 2*(data/255.0) - 1                                   # scales between -1 and 1
        data = torch.from_numpy(data).permute(2,0,1)                # reshape to tensor [3,32,32]
        return data, self.labels[index]

def save_images(images: torch.Tensor, path, epoch: int):
    # images: [batch_size x channels x height x width]

    grids = make_grid(images)
    narray = grids.permute(1,2,0).to('cpu')
    narray = narray.numpy()
    img = Image.fromarray(narray)
    img.save(f'{path}Epoch-{epoch}.png')