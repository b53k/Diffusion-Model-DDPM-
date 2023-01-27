# Diffusion-Model-DDPM
Rudimentary Implementation of Diffusion Models based on DDPM paper 2020 <br>

Diffusion-Model allows us to rapidly learn, sample from, and evaluate probabilities in deep generative models with thousands of layers/time-steps, as well as to compute conditional and posterior probabilities under the learned model. Diffusion process exists for any smooth target distribution, this method can capture data distributions of arbitrary form.

### Launch Training

```bash
python3 train.py --batch_size 32 --n-epochs 10000 --logdir logs --lr 3e-5
```
## Algorithm

The goal is to define a forward (or inference) diffusion process which converts any complex data distribution into a simple, tractable, distribution. Then learn a finite-time reversal of this diffusion process which defines the generative model distribution. 

#### Forward Trajectory

👉 Initial data distribution: $\large q(x^{(0)})$

👉 This is gradually converted to a well-behaved (analytically tractable) distribution $\pi(y)$ by repeated application of a Markov diffusion kernel $T_{\pi}(y\ |\ y^{'};\beta)$ where $\beta$ is diffusion rate

$$\begin{align}
\large \pi(y) &= \int T_\pi(y\ |\ y^{'};\beta)\ \ \pi(y^{'})\ \ dy^{'} \tag{1}\\
\large q\big(x^{t}|x^{(t-1)}\big) &= T_\pi\Big( x^{(t)}\ |\ x^{(t-1)};  \beta_t\Big) \tag{2}
\end{align}$$

Here; forward diffusion kernel is $T_\pi\Big( x^{(t)}\ |\ x^{(t-1)};  \beta_t\Big) = \mathcal{N}\Big(x^{(t)}; x^{(t-1)}\sqrt{1-\beta_t}, \ I\beta_t \Big)$ i.e. Gaussian but can be other distribution as well for example a Binomial distribution.

The forward trajectory, corresponding to starting at the initial data distribution and performing $T$ steps of diffusion is given by,
$$\large q\Big( x^{(0....T)}\Big) = q\big(x^{(0)}\big) \prod_{t=1}^T q\Big(x^{(t)}\ |\ x^{(t-1)}\Big) \tag{3}$$

The above process allows for the sampling of $\large x_t$ at any arbitrary time step $\large t$ in a closed form using reparameterization trick. $$\large x^{(t)} = \sqrt{1-\beta_t}\ \  \large x^{(t-1)} \ + \ \sqrt{\beta_t} \ \mathcal{N}(0,I)$$

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open('Jiraya.jpg')
img = img.resize(size = (128,128))
current_img = np.asarray(img)/255.0

def forward_diffusion(previous_img, beta, t):
	beta_t = beta[t]
	mean = previous_img * np.sqrt(1.0 - beta_t)
	sigma = np.sqrt(beta_t) # variance = beta
	# Generate sample from N(0,1) of prev img size and scale to new distribution.
	xt = mean + sigma * np.random.randn(*previous_img.shape)
	return xt

time_steps = 100
beta_start = 0.0001
beta_end = 0.05
beta = np.linspace(beta_start, beta_end, time_steps)

samples = []

for i in range(time_steps):
	current_img = forward_diffusion(previous_img = current_img, beta = beta, t = i)

	if i%20 == 0 or i == time_steps - 1:
		# convert to integer for display
		sample = (current_img.clip(0,1)*255.0).astype(np.uint8) 
		samples.append(sample)

  
plt.figure(figsize = (12,5))
for i in range(len(samples)):
	plt.subplot(1, len(samples), i+1)
	plt.imshow(samples[i])
	plt.title(f'Timestep: {i*20}')
	plt.axis('off')

plt.show()
```
![image](https://user-images.githubusercontent.com/91229701/215155134-af18baf2-1bdf-4aeb-b37d-bebca2e3c15c.png)

#### Reverse Trajectory

Generative distribution is trained to describe the above (forward) trajectory, but in the reverse direction,

$$\large \begin{align} 
\mathcal{p}\big(x^{(T)}\big) &= \pi\big(x^{(T)}\big) \tag{4} \\
\mathcal{p}\big(x^{(0...T)}\big) &= \mathcal{p}\big(x^{(T)}\big)\ \prod_{t=1}^T\mathcal{p}\Big(x^{(t-1)}\ |\  x^{(t)} \Big) \tag{5}
\end{align}$$

If we take $\large \beta_t$ to be small enough then, $q\big(x^{(t-1)}|x^{(t)}\big)$ will also be a Gaussian distribution. The longer the trajectory the smaller the diffusion rate $\large \beta$ can be made. During learning only the mean and the covariance for a Gaussian diffusion kernel need to be estimated. The functions defining the mean and the covariance of the reverse process are:
$$\large f_{\mu}(x^{(t)},t) \text{and} f_\Sigma(x^{(t)},t)$$

Practically speaking, we don't know $\large q\big(x^{t-1}|x^{(t)}\big)$ as it is intractable since the statistical estimate requires computations involving the data distribution. Therefore, we approximate $\large q\big(x^{t-1}|x^{(t)}\big)$ with a parameterized model $\large p_\theta$ (e.g. a neural network). With the parameterized model, we have

$$\large \mathcal{p}_\theta\big(x^{(0...T)}\big) = \mathcal{p}_\theta\big(x^{(T)}\big)\ \prod_{t=1}^T\mathcal{p}_\theta\Big(x^{(t-1)}\ |\  x^{(t)} \Big)$$

i.e. starting with the pure Gaussian noise, the model learns the joint distribution $\large \mathcal{p}_{\theta}(x^{(0:T)})$ . Conditioning the model on time-step t, it will learn to predict the Gaussian parameters for each time-step.

Here;
$$\large{p_\theta(x^{(t-1)}|x^{(t)})} = \mathcal{N}\Big(x^{(t-1)}; \ \ \mu_\theta(x^{(t)},t),\ \ \Sigma_\theta(x^{(t)},t)\Big)$$

### Training

The probability the generative model assigns to the data is

$$\large \mathcal{p}(x^{(0)}) = \int p(x^{(0...T)}) \ dx^{(1...T)} \tag{6}$$

This tells us that if we were to calculate $\large p(x^{(0)})$ we need to marginalize over all the possible trajectories to arrive at the initial distribution starting from the noise sample...which is intractable in practice. However, we can maximize a lower bound.

In #Diffusion-Model, the forward process is fixed and only reverse process needs to be trained i.e. only a single network is trained unlike #Variational-Auto-Encoder. 

#Diffusion-Models are trained by finding the reverse Markov transitions that maximize the likelihood of the training data. Similar to #Variational-Auto-Encoder training is based on minimizing the #Variational-Lower-Bound. Therefore, we optimize the negative log-likelihood.

$$\large \begin{align}
-\text{log}\ p_\theta(x_0) &\leq -\text{log}\ p_\theta(x_0) + D_{KL}\Big(\ q(x_{1:T}|x_0)\ \ ||\ \ p_\theta(x_{1:T}|x_0)\ \Big) \\
&= -\text{log}\ p_\theta(x_0) + \mathbb{E}_{x_{1:T}\ \sim\ q(x_{1:T}|x_0)}\Big[ \text{log}\ \frac{q(x_{1:T}|x_0)}{p_\theta(x_{1:T}|x_0)} \Big] \\
&= -\text{log}\ p_\theta(x_0) + \mathbb{E}_{x_{1:T}\ \sim\ q(x_{1:T}|x_0)}\Bigg[ \text{log}\ \frac{q(x_{1:T}|x_0)}{\frac{p_\theta(x_{1:T})\ p_\theta(x_0|x_{1:T})}{p_\theta(x_0)}} \Bigg] \\
&= \mathbb{E}_{x_{1:T}\ \sim\ q(x_{1:T}|x_0)}\Big[ \text{log}\ \frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:1:T})} \Big]
\end{align}$$

The above expression is Variational-Lower-Bound and

$$\large -\text{log}\ p_\theta(x_0) \leq \large \mathbb{E}_{x_{1:T}\ \sim \  q(x_{1:T}|x_0)}\Big[ \text{log}\ \frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:1:T})} \Big] \tag{7}$$

Now,

$$\large \begin{align}
\text{log}\ \frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:T})} &= \text{log}\ \frac{\prod_{t=1}^Tq\ (x_t|x_{t-1})}{p(x_T)\prod_{t=1}^T\ p_\theta(x_{t-1}|x_t)} \\
&= -\text{log}\ (p(x_T)) + \text{log}\ \frac{\prod_{t=1}^Tq\ (x_t|x_{t-1})}{\prod_{t=1}^T\ p_\theta(x_{t-1}|x_t)}\\
&= -\text{log}\ (p(x_T)) + \sum_{t=1}^T\text{log}\ \frac{q(x_t|x_{t-1})}{p_\theta(x_{t-1}|x_t)}\\
&= -\text{log}\ (p(x_T)) + \sum_{t=2}^T\text{log}\ \frac{q(x_t|x_{t-1})}{p_\theta(x_{t-1}|x_t)} + \text{log}\ \frac{q(x_1|x_0)}{p_\theta(x_0|x_1)}\\
&= -\text{log}\ (p(x_T)) + \sum_{t=2}^T\text{log}\ \frac{q(x_t|x_0)\ q(x_{t-1}|x_t,x_0)}{p_\theta(x_{t-1}|x_t)\ q(x_{t-1}|x_0)} + \text{log}\ \frac{q(x_1|x_0)}{p_\theta(x_0|x_1)}\\
&\because \text{Baye's Rule and conditoning on }x_0\text{ to avoid high variance}\\
&= -\text{log}\ (p(x_T)) + \sum_{t=2}^T\text{log}\ \frac{q(x_{t-1}|x_t,x_0)}{p_\theta(x_{t-1}|x_t)} + \sum_{t=2}^T\text{log}\ \frac{q(x_t|x_0)}{q(x_{t-1}|x_0)}{}+\text{log}\ \frac{q(x_1|x_0)}{p_\theta(x_0|x_1)}\\
&= -\text{log}\ (p(x_T)) + \sum_{t=2}^T\text{log}\ \frac{q(x_{t-1}|x_t,x_0)}{p_\theta(x_{t-1}|x_t)}\ +\ \text{log}\ \frac{q(x_T|x_0)}{q(x_1|x_0)} + \text{log}\ \frac{q(x_1|x_0)}{p_\theta(x_0|x_1)}\\
&= \text{log}\ \frac{q(x_T|x_0)}{p(x_T)} + \sum_{t=2}^T\text{log}\ \frac{q(x_{t-1}|x_t,x_0)}{p_\theta(x_{t-1}|x_t)}\ - \text{log}\ p_\theta(x_0|x_1)\\
&\because\ \text{Firt terms has no learnable parameter --- drop it}\\
\end{align}$$

From $(7)$;

$$\large \begin{align}
-\text{log}\ p_\theta(x_0) &\leq \large \mathbb{E}_{x_{1:T}\ \sim \  q(x_{1:T}|x_0)}\Big[ \text{log}\ \frac{q(x_{1:T}|x_0)}{p_\theta(x_{0:1:T})} \Big] \\
&= \large \mathbb{E}_{x_{1:T}\ \sim \  q(x_{1:T}|x_0)}\Big[ \sum_{t=2}^T\text{log}\ \frac{q(x_{t-1}|x_t,x_0)}{p_\theta(x_{t-1}|x_t)}\ - \text{log}\ p_\theta(x_0|x_1) \Big]\\
&= \large \mathbb{E}_{x_{1:T}\ \sim \  q(x_{1:T}|x_0)}\Big[ \sum_{t=2}^T\text{log}\ \frac{q(x_{t-1}|x_t,x_0)}{p_\theta(x_{t-1}|x_t)}\ - \text{log}\ p_\theta(x_0|x_1) \Big]\\
&= \sum_{t=2}^TD_{KL}\Big(q(x_{t-1}|x_t,x_0)\ ||\ p_\theta(x_{t-1}|x_t)\Big) -\text{log}\ p_\theta(x_0|x_1)
\end{align}$$

We can further simplify the first term above because;

$$\large
p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t,t), \beta_t I)\ \text{and}\ q(x_{t-1}|x_t,x_0) = \mathcal{N}(x_{t-1}; \ \tilde\mu_t(x_t,x_0), \tilde \beta_t I)$$

And,

$$\large \begin{align}
\tilde \mu_t(x_t,x_0) &:= \frac{\sqrt{\overline{\alpha}_{t-1}}\ \beta_t}{1 - \overline\alpha_t}\ x_0\ + \ \frac{\sqrt{\alpha_t}(1-\overline{\alpha}_{t-1})}{1-\overline\alpha_t}\ x_t \\
\tilde{\beta_t} &:= \frac{1-\overline{\alpha}_{t-1}}{1-\overline{\alpha}_t}\beta_t
\end{align}$$

where $\large \alpha_t = 1 - \beta_t$ and 
$\large \overline\alpha_t = {\prod_{t=s}}^t \alpha_s$ and further simplification as shown in <span style="color: magenta"><i>Denoising Diffusion Probabilistic Models [2020]</i></span> yields the loss function
$$\large \color{red}L_t \sim || \epsilon - \epsilon_\theta(x_t, t)||_2^2$$
where, $\large \epsilon := \mathcal{N}(0, I)$ and 
$\large (x_t,t)  = \big(\sqrt{\overline{\alpha}_t}\ x_0 + \sqrt{1-\overline{\alpha}_t}\ \epsilon,\ t\big)$



