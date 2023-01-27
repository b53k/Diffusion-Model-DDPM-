'''Training script Diffusion Model -- Very Basic Implementation :)'''
import torch, os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from misc.utils import Dataset, save_images
from model.diffusion import Diffusion
from model.unet import UNet
import argparse
import wandb
from tqdm import tqdm
import time

# Parse Argument
parser = argparse.ArgumentParser(description = 'Diffusion Trainer')
parser.add_argument('--batch-size', type = int, default = 128, metavar = 'N', help = 'input batch size for training (default: 128)')
parser.add_argument('--n-epochs', type = int, default = 100, metavar = 'N', help = 'input number of epochs for training (default: 100)')
parser.add_argument('--lr', type = float, default = 2e-5, metavar = 'N', help = 'input learning rate (default: 2e-5)')
parser.add_argument('--noreload', action = 'store_true', help = 'previously saved model will not be reloaded')
parser.add_argument('--logdir', type = str, help = 'Directory where results are logged')
args = parser.parse_args()

batch_size = args.batch_size
epochs = args.n_epochs
lr = args.lr


# Initialize wandb
wandb.init(project = 'DDPM', entity = 'b53k')

# Set Device and Seed (for reproducibility)
cuda_bool = False
if cuda_bool:
    try:
        device = torch.device('cuda')
    except:
        print ('Could not use CUDA device...defaulting to CPU')
        device = torch.device('cpu')
else:
    device = torch.device('cpu')

torch.manual_seed(2023)

# For numeric stability
torch.backends.cudnn.benchmark = True

# Dataset
path = './Linnaeus_Flower/'
ds = Dataset(path, img_size = 32)                                                     

# DataLoader
dataloader = DataLoader(dataset = ds, batch_size = batch_size, shuffle = True, drop_last = False)        

# Model
net = UNet().to(device)
diff = Diffusion(model = net, device = cuda_bool)

# Optimizer
optimizer = torch.optim.AdamW(net.parameters(), lr = lr)                                   

# Loss
l2_loss = torch.nn.MSELoss()

# Save and Load Checkpoint
root_path = './checkpoint/'


def save_checkpoint(name):
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    torch.save({
        'Epoch':epoch,
        'Model_State_Dict':net.state_dict(),
        'Optimizer_State_Dict':optimizer.state_dict(),
        'Loss':loss
    }, f'{root_path}Model-{name}.pkl')

def load_checkpoint(name):
    checkpoint = torch.load(f'{root_path}Model-{name}.pkl')
    net.load_state_dict(checkpoint['Model_State_Dict'])
    optimizer.load_state_dict(checkpoint['Optimizer_State_Dict'])
    epoch = checkpoint['Epoch']
    loss = checkpoint['Loss']

    print ('')
    print ('Checkpoint Loaded with the following parameters:')
    print ('Epoch:{} Loss: {:.4f}'.format(epoch,loss))
    print ('==============================================================================')
    print ('')

# Log results during training
sample_img_dir = os.path.join(args.logdir, 'sample_imgs/')
if not os.path.exists(sample_img_dir):
    os.makedirs(sample_img_dir)


# Add argument for noreload
reload_file = os.path.join(root_path, 'Model-latest_weights.pkl')
if not args.noreload and os.path.exists(reload_file):
    load_checkpoint('latest')

# Training Loop

net.train()
wandb.watch(models = net, criterion = None, log = 'all')
pbar = tqdm(total = epochs, position = 0, leave = True)

for epoch in range(epochs):
    running_loss = 0.0
    for index, (image_batch, _) in enumerate(dataloader):

        image_batch = image_batch.to(device)

        # Sample timesteps
        t = diff.sample_timesteps(n = image_batch.size(dim=0))
        
        # Get noisy image and actual noise                                                  
        noisy_img, real_noise = diff.add_noise(image_batch, t)
        noisy_img = noisy_img.float()
        real_noise = real_noise.float()

        # Obtained predicted noise via parameterized model (U-Net)
        predicted_noise = net(noisy_img, t.float())

        # Calculate L2 loss
        loss = l2_loss(real_noise, predicted_noise)

        # Clear gradients
        optimizer.zero_grad()

        # Backpropagate
        loss.backward()

        # Take a small step in negative gradient direction
        optimizer.step()

        # Cumulative loss
        running_loss += loss.item()

    if epoch%10 == 0 or epoch == epochs-1:
        # Sample 4 images and see how they look like during training
        print ('')
        print ('Sampling Images...')
        start = time.time()
        sample_images = diff.sample_img(n = 4)
        save_images(images = sample_images, path = sample_img_dir, epoch = epoch)
        end = time.time()
        hr, rem = divmod(end-start, 3600)
        minutes, sec = divmod(rem, 60)
        print ('Finished Sampling in {:0>2}:{:0>2}:{:05.2f}'.format(int(hr), int(minutes), sec))                                          

        # save Checkpoint
        print ('')
        print ('==============================================================================')
        print ('Saving Checkpoint with the following parameters:')
        print ('Epoch: {} Loss: {:.4f}'.format(epoch, running_loss/image_batch.size(dim=0)))
        print ('==============================================================================')
        print ('')
        save_checkpoint(name = 'latest_weights') # Overrides  

    # Log Training Process
    wandb.log({'epoch': epoch+1, 'batch_loss': running_loss/image_batch.size(dim=0)})      

    pbar.set_postfix_str({'Batch Loss': '{0:.4f}'.format(running_loss/image_batch.size(dim=0))})
    pbar.update()
pbar.close()

print ('')
print ('Finished Training')
print ('')