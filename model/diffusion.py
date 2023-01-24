import torch
import torch.nn as nn

class Diffusion:
    def __init__(self, model: nn.Module, time_steps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02, image_size: int = 32):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.time_steps = time_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.image_size = image_size
        
        self.beta = (torch.linspace(self.beta_start, self.beta_end, self.time_steps)).to(self.device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim = 0)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor):
        ''' Forward process....Samples from q(x_t | x_0)
            x0: [batch_size x channels x height x width]
             t: [batch_size]
        '''
        ϵ = torch.randn_like(x0).to(self.device)
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t])                                                      # TODO: Check Dimension
        mean = sqrt_alpha_bar * x0
        var = (torch.sqrt(1-self.alpha_bar[t])) * ϵ
        
        return mean + var, ϵ
    
    def sample_img(self, n: int):
        ''' Reverse process....Samples from p0(x_t_1 | x_t)
             n: number of samples to generate
             produces noiseless image given a noisy image
        '''
        self.model.eval()
        with torch.no_grad():
            x = torch.randn(n, 3, self.image_size, self.image_size).to(self.device)                                                                            
            for t in reversed(range(1, self.time_steps)):

                if t > 1:
                    z = torch.randn_like(x).to(self.device)                                                                 
                else:
                    z = torch.zeros_like(x).to(self.device)                                                                

                alpha_t = self.alpha[t]                                                                     
                alpha_bar_t = self.alpha_bar[t]
                sigma_t = torch.sqrt(self.beta[t])
                coeff = (1 - alpha_t)/(torch.sqrt(1 - alpha_bar_t))
                curr_t_step = torch.ones((n)).long()*t
                curr_t_step = curr_t_step.to(self.device)                                                      
                pred_noise = self.model(x, curr_t_step)

                x = (1/torch.sqrt(alpha_t))*(x - coeff*pred_noise) + sigma_t*z
        self.model.train()
        x = (x.clamp(-1,1)+1)/2
        x = (x*255.0).type(torch.uint8)

        return x

'''

device = torch.device('cuda')
img = torch.rand(10,3,8,8).to(device)
t = torch.randint(1,100,(1,)).to(device)

from unet import UNet
net = UNet().to(device)
diff = Diffusion(model = net)
res = diff.sample_img(1)
res = res[0].permute(1,2,0)

'''