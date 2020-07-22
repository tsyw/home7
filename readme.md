# README

这篇文章中介绍了一种采用生成对抗网络的新的方案来解决图中的半监督学习。首先是我的阅读笔记，然后试探性地“复现”。

## Link

[Semi-supervised Learning on Graphs with Generative Adversarial Nets](https://arxiv.org/abs/1809.00130v1)

### Motivation

Using Generative Adversarial Nets(GAN) to help semi-supervised learning on graphs.

### Idea

Using GAN to generate fake nodes for low gap density area.

#### GANs

$$\min\limits_{G}\max\limits_{D}V(G,D)=E_{x\sim p_d(x)}\log D(x)+E_{z\sim p_z(z)}\log (1-D(G(z)))$$

This formula means that we want to minimize the Generated graphs(fake) and maximize the real graphs for D function. At the same time, we should maximize the second term for G function to fool the D function.

#### Gap density

Try to weaken the effect of propagation across density gaps by fake nodes.

Problem: GAN cannot use the graphs' topology.

Solution: Use some embedding techniques such as DeepWalk, Line, NetMF and so on to learn the latent distributed representation. After the graph topology learning, we can use these infomation to train GANs.

#### General techniques

1. Batch Normalization (BN) for gradient stability

2. Weight Normalization for trainable weight scale

3. additive Gaussian noise in D function (classifier in this paper) for training

#### Architecture

Classifier(D function): softmax with an additional fake class
