# README

这篇文章中介绍了一种采用生成对抗网络的新的方案来解决图中的半监督学习。首先是我的阅读笔记，然后试探性地“复现”。

## Link

[Semi-supervised Learning on Graphs with Generative Adversarial Nets](https://arxiv.org/abs/1809.00130v1)

### Motivation

Using Generative Adversarial Nets(GAN) to help semi-supervised learning on graphs. The curse of dimensionality may make the propagation across different cluster easier, so authors want to generate more fake nodes between different clusters. 

### Idea

Using GAN to generate fake nodes for low gap density area.

#### GANs

$$\min\limits_{G}\max\limits_{D}V(G,D)=E_{x\sim p_d(x)}\log D(x)+E_{z\sim p_z(z)}\log (1-D(G(z)))$$

This formula means that we want to minimize the Generated graphs(fake) and maximize the real graphs for D function. At the same time, we should maximize the second term for G function to fool the D function.

#### Gap density

Try to weaken the effect of propagation across density gaps by adding fake nodes.

Problem: GAN cannot use the graphs' topology, the G function cannot generate fake nodes in low density areas.

Solution: Use some embedding techniques such as DeepWalk, Line, NetMF and so on to learn the latent distributed representation. After the graph topology learning, we can use these infomation to train GANs.

#### General techniques

1. Batch Normalization (BN) for gradient stability

2. Weight Normalization for trainable weight scale

3. additive Gaussian noise in D function (classifier in this paper) for training

4. neighbor fusion

#### Architecture

Classifier(D function): softmax with an additional fake class

Loss function:

1. D function:$L_D =loss_{sup} + \lambda_0 loss_{un} + \lambda_1 loss_{ent} + loss_{pt}$
   1. for labeled nodes: maximize cross entropy
   2. for classify: should not be mapped into the low density area and fake nodes should be mapped into the low density area: minimize p(M|x),maximize p(M|g(z))
   3. For unlabeled nodes: maximize the entropy over definite labels
   4. For low density area: widen the gap: maximize the cosine distance

2. G function:$L_G =loss_{fm} + \lambda_2 loss_{pt}$
   1. Generated nodes: should be in the low density area: minimize the distance to central point
   2. Generated samples should not overfit at the only center point: the same as 4 in D function.

3. hyper-parameter: to control the relative strength of different terms.