## Link

[Semi-supervised Learning on Graphs with Generative Adversarial Nets](https://arxiv.org/abs/1809.00130v1)

### Motivation

Using Generative Adversarial Nets(GAN) to help semi-supervised learning on graphs.

### Idea

Using GAN to generate fake nodes for low gap density area.

#### GANs

$$\min\limits_{G}\max\limits_{D}V(G,D)=E_{x\sim p_d(x)}\log D(x)+E_{z\sim p_z(z)}\log (1-D(G(z)))$$

This formula means that we want to minimize the Generated graphs(fake) and maximize the real graphs for D function. At the same time, we should maximize the second term for G function to fool the D function.

