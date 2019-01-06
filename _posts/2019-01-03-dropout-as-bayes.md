---
layout: post
title: "Notes on: Dropout as a Bayesian Approximation"
date: 2019-01-03
mathjax: true
---

Original Paper by: Yarin Gal, Zoubin Ghahramani

NB: There exists a [suplementary appendix/paper](https://arxiv.org/abs/1506.02157.pdf) to this paper which is helpful for background.

# High Level Overview

* This paper shows that dropout NNs are simply approximations to deep GPs, and that the network architecture (i.e., layout, activations, etc.) are simply design choices in the covariance function.
* In short, dropout is simply a variational approach to approximating a deep GP, with the Bernoulli random variables over activations injecting the stochasticity needed.
* The resultant predictions (and uncertainty estimates) are obtained by retaining the dropout variables in test time, and gives state-of-the-art in both prediction and uncertainty quality.

# Introduction

This paper motivates the need for uncertainty estimates by elucidating on the point that classification/softmax outputs are NOT exactly confidences, which is illustrated in its Fig 1. In short, if one was to pass a single sample of an output through a softmax, it is possible that the output could be as high as ~1.0. However, passing the entire distribution over that output could produce a range for which values as low as 0.4 are likely (in the softmax output distribution). Evidently such a model may be misleading in decision based processes, as the former metric could be interpreted as high confidence, when in fact the model has low confidence.

The paper notes that in traditional dropout, the binary distribution over the activation of weights is removed during test time, whereas in fact maintaining this distribution allows for Bayesian analyses of the modelling process. The paper makes this link explicit by showing that a neural network with dropout is simply an approximation to a Deep Gaussian Process[^1] (i.e. nested GPs).

# Related Research

Compared with the [Bayes-by-Backprop paper](/posts/2018/12/06/bayes-backprop.html) there is less computational cost as they don't need to double the parameters to account for uncertainty (i.e. mean _and_ variance). Furthermore, dropout is part of the standard state of the art toolkit, so we can effectively get a Bayesian neural network for 'free'.

# Dropout as a Bayesian Approximation

The dropout objective with L2 regularisation results in a variational approximation to a deep GP's posterior, regardless of activiation function and depth. The standard objective function to minimise (i.e., cost) is:

$$
\begin{equation}
\mathcal{L}_{\text{dropout}} := \frac{1}{N}\sum^N_{i=1}E(\mathbf{y}_i,\hat{\mathbf{y}}_i) + \lambda \sum^L_{i=1}(||\mathbf{W}_i||^2_2 + ||\mathbf{b}_i||^2_2).\label{eq:NNLoss}
\end{equation}
$$

We sample a new set of binary variables $$\mathbf{z}_i$$ (which determine the activation of the **nodes** per layer) for **each** datapoint indexed by $$i$$. Note that we parameterise this using variables $$p_l$$ which indicate the probability of the nodes being active in layer $$l$$.

In contrast, we write the covariance function of a standard GP as follows (note that this is a valid covariance function which is parameterised by a single hidden layer neural network):

$$
\begin{equation}
\mathbf{K}(\mathbf{x},\mathbf{y}) = \int p(\mathbf{w})p(b)\sigma(\mathbf{w}^\top\mathbf{x} + b)\sigma(\mathbf{w}^\top\mathbf{y} + b)\text{d}\mathbf{w}\text{d}b
\end{equation}
$$

where $$\sigma$$ represents the non-linearity (i.e., ReLU, tanh). As a consequence, changing the non-linearity modifies the covariance function of the resultant GP. Finally, we note $$p(\mathbf{w})$$ is a standard multivariate normal. This is approximated as:

$$
\begin{equation}
\hat{\mathbf{K}}(\mathbf{x},\mathbf{y}) = \frac{1}{K}\sum^K_{k=1} \sigma(\mathbf{w}_k^\top\mathbf{x} + b_k)\sigma(\mathbf{w}_k^\top\mathbf{y} + b_k)
\end{equation}
$$

where $$K$$ is the number of hidden units in a single hidden layer NN (single since we are approximating a GP not a deep GP for now). With this insight, we observe that we can write the parameters of the covariance function as follows:

$$
\begin{equation}
\mathbf{W}_1 = [\mathbf{w}_k]^K_1, \quad\mathbf{b} = [b_k]^K_1
\end{equation}
$$

which makes the link to the NN more explicit. We complete the notation by writing part of the generative model of the GP (recalling the 0 mean over the prior doesn't restrict the posterior):

$$
\begin{align}
\mathbf{F}|\mathbf{X},\mathbf{W}_1,\mathbf{b} & \sim \mathcal{N}(0,\hat{\mathbf{K}}(\mathbf{X},\mathbf{X})) \\
\mathbf{Y} | \mathbf{F} & \sim \mathcal{N}(\mathbf{F},\tau^{-1}\mathbf{I}_N).
\end{align}
$$

This gives the following marginal:

$$
\begin{equation}
P(\mathbf{Y}|\mathbf{X}) = \int P(\mathbf{Y}|\mathbf{F})P(\mathbf{F}|\mathbf{X},\mathbf{W}_1,\mathbf{b})p(\mathbf{W}_1)p(\mathbf{b})\text{d}\mathbf{F}\text{d}\mathbf{W}_1\text{d}\mathbf{b}.\label{eq:marginal}
\end{equation}
$$

We can rewrite the covariance function in more traditional kernel notation, recalling each row is $$1\times K$$:

$$
\begin{equation}
\phi(\mathbf{x},\mathbf{W}_1,\mathbf{b}) = \sqrt{\frac{1}{K}}\sigma(\mathbf{W}_1^\top\mathbf{x}+\mathbf{b})
\end{equation}
$$

and putting these in a matrix such that $$\Phi = [\phi(\mathbf{x}_n,\mathbf{W}_1,\mathbf{b})]^N_{n=1}$$ and $$\hat{\mathbf{K}}(\mathbf{X},\mathbf{X}) = \Phi\Phi^{\top}$$, we can rewrite the marginal Eq \ref{eq:marginal} as (integrating out $$\mathbf{F}$$):

$$
\begin{equation}
P(\mathbf{Y}|\mathbf{X}) = \int \mathcal{N}(\mathbf{Y},\Phi\Phi^{\top} + \tau^{-1}\mathbf{I}_N)p(\mathbf{W}_1)p(\mathbf{b})\text{d}\mathbf{W}_1\text{d}\mathbf{b}.\label{eq:marginal2}
\end{equation}
$$

At this point the paper introduces an auxiliary random variable $$\mathbf{w}_d$$ to represent the set of weights mapping the hidden layer to the output layer by rewriting the $$\mathbf{Y}$$ distribution integrand in Eq \ref{eq:marginal2} as joint over all the individual output dimensions (since we assume these are independent in a multi-output Gaussian process). We trivially obtain the orginal distribution by integrating over this random variable:

$$
\begin{equation}
\mathcal{N}(\mathbf{y}_d;0,\Phi\Phi^{\top}+\tau^{-1}\mathbf{I}_N) = \int \mathcal{N}(\mathbf{y}_d;\Phi\mathbf{w}_d,\tau^{-1}\mathbf{I}_N)\mathcal{N}(\mathbf{w}_d;0,\mathbf{I}_K)\text{d}\mathbf{w}_d.
\end{equation}
$$

Writing $$\mathbf{W}_2 = [\mathbf{w}_d]^D_{d=1}$$ (a $$K \times D$$ matrix), we get the following form for the marginal in Eq \ref{eq:marginal2}:

$$
\begin{equation}
P(\mathbf{Y}|\mathbf{X}) = \int P(\mathbf{Y}|\mathbf{X},\mathbf{W}_1,\mathbf{W}_2,\mathbf{b})p(\mathbf{W}_1)p(\mathbf{W}_2)p(\mathbf{b})\text{d}\mathbf{W}_1\text{d}\mathbf{W}_2\text{d}\mathbf{b}.\label{eq:marginal3}
\end{equation}
$$

The paper notes that this is equivalent to "the weighted basis function interpretation of the Gaussian process" (**need to look into this**). It is readily apparent that the Bernoullis over the weights due to dropout are able to create some sort of distribution over them.

Having parameterised the GP with the weights and biases of a neural network, we can perform variational inference to estimate the posteriors over these parameters. As standard, we create an approximating distribution, and assume a tractable form:

$$
\begin{equation}
q(\mathbf{W}_1,\mathbf{W}_2,\mathbf{b}) := q(\mathbf{W}_1)q(\mathbf{W}_2)q(\mathbf{b}).
\end{equation}
$$

We assume the prior over the weights to be a factorised (over the input dimension $$Q$$ and hidden dimension $$K$$ for for $$\mathbf{W}_1$$ and $$\mathbf{W}_2$$ respectively) Gaussian mixture with two components; one centred at 0 (i.e. turned off) and another with mean $$\mathbf{m}_q \in \mathbb{R}^K$$ for $$\mathbf{W}_1$$ for example.

$$
\begin{align}
q(\mathbf{W}_1) &= \prod^Q_{q=1} q(\mathbf{w}_q)\label{eq:paramstart}\\
q(\mathbf{w}_q) &= p_1\mathcal{N}(\mathbf{m}_q,\sigma^2\mathbf{I}_K) + (1-p_1)\mathcal{N}(\mathbf{0},\sigma^2\mathbf{I}_K)\\
q(\mathbf{W}_2) &= \prod^K_{k=1} q(\mathbf{w}_k)\\
q(\mathbf{w}_k) &= p_2\mathcal{N}(\mathbf{m}_k,\sigma^2\mathbf{I}_D) + (1-p_2)\mathcal{N}(\mathbf{0},\sigma^2\mathbf{I}_D).\label{eq:paramend}
\end{align}
$$

The prior over $$\mathbf{b}$$ is simply a diagonal Gaussian:

$$
\begin{equation}
q(\mathbf{b}) = \mathcal{N}(\mathbf{m},\sigma^2\mathbf{I}_K).
\end{equation}
$$

Having parameterised the weights/biases with the variational parameters $$\mathbf{M}_1=[\mathbf{m}_q]^Q_{q=1},\mathbf{M}_2=[\mathbf{m}_k]^K_{k=1},\mathbf{m}$$, we can minimise the variational objective with respect to these as per usual. We can derive the variational objective as follows, by recalling the posterior distribution over an output $$\mathbf{y}^{*}$$ given an input point $$\mathbf{x}^{*}$$:

$$
\begin{equation}
p(\mathbf{y}^*|\mathbf{x}^*,\mathbf{X},\mathbf{Y}) = \int p(\mathbf{y}^*|\mathbf{x}^*,\omega)p(\omega|\mathbf{X},\mathbf{Y}) \text{d}\omega.
\end{equation}
$$

Since the posterior over the parameters $$p(\omega\vert\mathbf{X},\mathbf{Y})$$ is usually intractable (due to the partition function), we approximate it with some function $$q(\omega)$$. We then minimise the KL-divergence between this and the true posterior, i.e. ($\text{D}_\text{KL}(q(\omega)\vert\vert p(\omega\vert\mathbf{X},\mathbf{Y}))$). We can then approximate the posterior over the test outputs:

$$
\begin{equation}
q(\mathbf{y}^*|\mathbf{x}^*) = \int p(\mathbf{y}^*|\mathbf{x}^*,\omega)q(\omega) \text{d}\omega.
\end{equation}
$$

To obtain the variational objective, recalling we wish to minimise the KL-divergence between the approximation and the true posterior, we obtain:

$$
\begin{align}
\text{D}_\text{KL}(q(\omega)|p(\omega|\mathbf{X},\mathbf{Y})) &= \mathbb{E}_q\left[ \log \frac{q(\omega)}{p(\omega|\mathbf{X},\mathbf{Y})} \right]\\
&= \mathbb{E}_q\left[ \log q(\omega) - \log p(\omega|\mathbf{X},\mathbf{Y}) \right]\\
&= \mathbb{E}_q\left[ \log q(\omega) - \log p(\omega,\mathbf{Y}|\mathbf{X}) + \log p(\mathbf{Y} | \mathbf{X}) \right]\\
&= \mathbb{E}_q\left[ \log q(\omega) - \log p(\omega,\mathbf{Y}|\mathbf{X}) \right] + \log p(\mathbf{Y} | \mathbf{X}).
\end{align}
$$

Rearranging yields:

$$
\begin{align}
\log p(\mathbf{Y} | \mathbf{X}) - \text{D}_\text{KL}(q(\omega)|p(\omega|\mathbf{X},\mathbf{Y})) &= - \mathbb{E}_q\left[ \log q(\omega) - \log p(\omega,\mathbf{Y}|\mathbf{X}) \right]\\
&= \mathbb{E}_q\left[  -\log q(\omega) + \log p(\mathbf{Y}|\mathbf{X},\omega) + \log p(\omega) \right]\\
&= \int q(\omega)\log p(\mathbf{Y}|\mathbf{X},\omega)\text{d}\omega - \text{D}_\text{KL}(q(\omega)|p(\omega))
\end{align}
$$

as per Eq 7 in the paper. Incorporating our parameterisation so far, we get:

$$
\begin{align}
 \mathcal{L}_\text{VI} &= \int q(\mathbf{W}_1,\mathbf{W}_2,\mathbf{b})\log p(\mathbf{Y}|\mathbf{X},\mathbf{W}_1,\mathbf{W}_2,\mathbf{b}) - \text{D}_\text{KL}(q(\mathbf{W}_1,\mathbf{W}_2,\mathbf{b})|p(\mathbf{W}_1,\mathbf{W}_2,\mathbf{b})) \\
  &= \sum^N_{n=1} \int q(\mathbf{W}_1,\mathbf{W}_2,\mathbf{b})\log p(\mathbf{y}_n|\mathbf{x}_n,\mathbf{W}_1,\mathbf{W}_2,\mathbf{b}) - \text{D}_\text{KL}(q(\mathbf{W}_1,\mathbf{W}_2,\mathbf{b})|p(\mathbf{W}_1,\mathbf{W}_2,\mathbf{b}))\label{eq:VI}
\end{align}
$$

where Eq \ref{eq:VI} follows due to independence assumptions on the output space and linear algebraic identities (see Appendix 3.3).

Recall however that the actual variational parameters are not $$\mathbf{W}_1,\mathbf{W}_2,\mathbf{b}$$ but $$\mathbf{M}_1,\mathbf{M}_2,\mathbf{m}$$, which are the weights before dropout. We therefore need to write one in terms of the other, recalling Eqs \ref{eq:paramstart}-\ref{eq:paramend}:

$$
\begin{align}
\mathbf{W}_1 &= \mathbf{z}_1(\mathbf{M}_1+\sigma \epsilon_1) + (1-\mathbf{z}_1)\sigma \epsilon_1\\
\mathbf{W}_2 &= \mathbf{z}_2(\mathbf{M}_2+\sigma \epsilon_2) + (1-\mathbf{z}_2)\sigma \epsilon_2\\
\mathbf{b} &= \mathbf{m} + \sigma \epsilon.
\end{align}
$$

This notation allows us to 'sample' our $$W$$ terms as $$\hat{W}$$, and apply Monte Carlo integration:

$$
\begin{align}
 \mathcal{L}_\text{MC} &= \sum^N_{n=1} \log p(\mathbf{y}_n|\mathbf{x}_n,\hat{\mathbf{W}}^n_1,\hat{\mathbf{W}}^n_2,\hat{\mathbf{b}}^n) - \text{D}_\text{KL}(q(\mathbf{W}_1,\mathbf{W}_2,\mathbf{b})|p(\mathbf{W}_1,\mathbf{W}_2,\mathbf{b})).\label{eq:KLMC}
\end{align}
$$

This leaves the KL-divergence. Since we assume the variables in the joint to be independent, it is trivial to show that the resultant KL-divergence is simply the sum of all the constituents:

$$
\begin{align}
\text{D}_\text{KL}(q(\mathbf{W}_1,\mathbf{W}_2,\mathbf{b}) || p(\mathbf{W}_1,\mathbf{W}_2,\mathbf{b})) &= \text{D}_\text{KL}(q(\mathbf{W}_1)q(\mathbf{W}_2)q(\mathbf{b}) || p(\mathbf{W}_1)p(\mathbf{W}_2)p(\mathbf{b}))\\
&= \int q(\mathbf{W}_1)q(\mathbf{W}_2)q(\mathbf{b}) \log\frac{q(\mathbf{W}_1)q(\mathbf{W}_2)q(\mathbf{b})}{p(\mathbf{W}_1)p(\mathbf{W}_2)p(\mathbf{b})}\\
&= \int q(\mathbf{W}_1)q(\mathbf{W}_2)q(\mathbf{b}) \left[\log\frac{q(\mathbf{W}_1)}{p(\mathbf{W}_1)} + \log\frac{q(\mathbf{W}_2)}{p(\mathbf{W}_2)} + \log\frac{q(\mathbf{b})}{p(\mathbf{b})} \right]\\
&= \text{D}_\text{KL}(q(\mathbf{W}_1)) || p(\mathbf{W}_1)) + \text{D}_\text{KL}(q(\mathbf{W}_2)) || p(\mathbf{W}_2)) + \text{D}_\text{KL}(q(\mathbf{b})) || p(\mathbf{b}))
\end{align}
$$

With this in mind, and following the appendix Proposition 1, the KL-divergence can be approximated (assuming small $$\sigma^2$$) as follows for the weight, and analytically for the bias terms respectively:

$$
\begin{align}
\text{D}_\text{KL}(q(\mathbf{W}_1)) || p(\mathbf{W}_1)) &\approx QK(\sigma^2 - \log(\sigma^2)-1) + \frac{p_1}{2}\sum^Q_{q=1}\mathbf{m_q}^\top\mathbf{m_q} + C\\
\text{D}_\text{KL}(q(\mathbf{b})) || p(\mathbf{b})) &= \frac{1}{2}(\mathbf{m}^\top\mathbf{m} + K(\sigma^2 + \log(\sigma^2) - 1)) + C
\end{align}
$$

Finally, shoving all the terms back into Eq \ref{eq:KLMC}, and ignoring all terms which go to 0 when we differentiate (i.e., $$\sigma,\tau$$) we get:

$$
\begin{align}
\mathcal{L}_\text{MC} &\propto -\frac{\tau}{2}\sum^N_{n=1}||\mathbf{y}_n - \hat{\mathbf{y}}_n||^2_2 - \frac{p_1}{2}||\mathbf{M}_1||^2_2 - \frac{p_2}{2}||\mathbf{M}_2||^2_2 - \frac{1}{2}||\mathbf{m}||^2_2\\
&\propto -\frac{1}{2N}\sum^N_{n=1}||\mathbf{y}_n - \hat{\mathbf{y}}_n||^2_2 - \frac{p_1}{2\tau N}||\mathbf{M}_1||^2_2 - \frac{p_2}{2\tau N}||\mathbf{M}_2||^2_2 - \frac{1}{2\tau N}||\mathbf{m}||^2_2 \label{eq:MCEnd}
\end{align}
$$

where the final KL terms are simplified notation of the L2 norms. By setting the original NN loss appropriately, we can show that Eq \ref{eq:MCEnd} and Eq \ref{eq:NNLoss} are the same. This shows that dropout is essentially variational inference over an approximation of a deep GP.

# Obtaining Model Uncertainty

This is done by estimating the sufficient measures for variance, namely the first and second moments of the output distribution. To obtain the first moment, we sample the $$z$$ vectors $$T$$ times, and take the average over the output:

$$
\begin{equation}
\mathbb{E}_{q(\mathbf{y}^*|\mathbf{x}^*)}[\mathbf{y}^*] \approx \frac{1}{T} \sum^T_{t=1} \hat{\mathbf{y}}^*(\mathbf{x}^*,\mathbf{W}^t_1,...,\mathbf{W}^t_L).
\end{equation}
$$

For the second moment, the same follows:

$$
\begin{equation}
\mathbb{E}_{q(\mathbf{y}^*|\mathbf{x}^*)}[(\mathbf{y}^*)^\top(\mathbf{y}^*)] \approx \tau^{-1}\mathbf{I}_D + \frac{1}{T} \sum^T_{t=1} \hat{\mathbf{y}}^*(\mathbf{x}^*,\mathbf{W}^t_1,...,\mathbf{W}^t_L)^\top \hat{\mathbf{y}}^*(\mathbf{x}^*,\mathbf{W}^t_1,...,\mathbf{W}^t_L).
\end{equation}
$$

This gives a variance of:

$$
\begin{equation}
\text{Var}_{q(\mathbf{y}^*|\mathbf{x}^*)}[(\mathbf{y}^*)] \approx \tau^{-1}\mathbf{I}_D + \frac{1}{T} \sum^T_{t=1} \hat{\mathbf{y}}^*(\mathbf{x}^*,\mathbf{W}^t_1,...,\mathbf{W}^t_L)^\top \hat{\mathbf{y}}^*(\mathbf{x}^*,\mathbf{W}^t_1,...,\mathbf{W}^t_L) - \mathbb{E}_{q(\mathbf{y}^*|\mathbf{x}^*)}[\mathbf{y}^*]^\top \mathbb{E}_{q(\mathbf{y}^*|\mathbf{x}^*)}[\mathbf{y}^*]
\end{equation}
$$

Noting the equivalences between equations \ref{eq:NNLoss} and \ref{eq:MCEnd}, we get the following identity for the precision:

$$
\begin{equation}
\tau = \frac{pl^2}{2N\lambda}
\end{equation}
$$

thus, combined with the variance estimate, we see that variance is simply that of the dropout network with dropout still active during test time. The paper notes that if forward passes are parelellised, then running time is equivalent to a normal NN.

# Experiments

The Bayesian methods behave as expected, producing higher uncertainty away from the data in the case of ReLU activations for example. Interestingly, TanH activations don't exhibit this, and are more confident away from the data than would be expected. This is attributed to the saturation of inputs in the networks.

What is most of interest is how well dropout networks estimate uncertainty; whilst mathematically the paper shows that dropout nets approximate GPs, and we have an analytical expression for variance over the outputs, it remains to be seen how good these approximations are in practice.

The way this uncertainty quality is assessed is using log-likelihoods over the testing outputs given testing inputs:

$$
\begin{align}
\log p(\mathbf{y}^*|\mathbf{x}^*,\mathbf{X},\mathbf{Y}) &\approx \log \frac{1}{T}\sum^T_{t=1} p(\mathbf{y}^*|\mathbf{x}^*,\boldsymbol{\omega}_t)\\
&= \text{logsumexp}\left( -\frac{1}{2}\tau||\mathbf{y}-\hat{\mathbf{y}}_t||^2 \right) -\log T - \frac{1}{2}\log 2\pi - \frac{1}{2}\log \tau^{-1}\label{eq:regll}
\end{align}
$$

with Eq \ref{eq:regll} holding true for regression tasks. Selecting the value for $$\tau$$ is done using Bayesian Optimisation over validation data to find the maximum log-likehood, and observing Eq \ref{eq:regll} we note its tradeoff; being too uncertain (i.e., small precision $$\tau$$) will cause the (negative) last term to be large, resulting in smaller log-likehoods. Conversely, overconfident predictions (i.e., large $$\tau$$) will cause any errors in the prediction (i.e., the first term) to become amplified.

In short, this log-likehood of the test data can be used as a proxy for uncertainty predictions, since overconfidence would be punished, and vice-versa. 

* AS AN ASIDE: However, it does feel a bit 'cheap' since it's non-independent of model accuracy; therefore given dropout perhaps performs the best out of all of these methods on a pure data-fitting front, it's as if its log-likehood has an unfair 'head-start'.

The paper then demonstrates that dropout provides state-of-the-art performance in this regard, and could perhaps be improved by specific tuning. Looking at [Yarin Gal's PhD Thesis](http://www.cs.ox.ac.uk/people/yarin.gal/website/thesis/4_uncertainty_quality.pdf) however, it is clear that it still doesn't estimate uncertainties as well as classic GPs, likely due to the approximations made in the derivations.

Finally there is some exposition into RL, where it is noted that convergence occurs quicker, but $$\epsilon$$-greedy schemes achieve higher final return as pure exploitation occurs, whereas this is not the case in the dropout network.

# Conculsion

It can be shown that the existing dropout methods are simply approximations to deep GPs, and furthermore we can get a posterior distribution over the predictions (given the training data and test input point) for free.

Furthermore, the analogies between dropout NNs and GPs make it clear that the Bernoulli distributions are simply approximations we make in the variational distribution, and that perhaps more suitable choices exist (i.e., more accurate models of the posterior).

Weight decay and activation functions in the nodes are also nothing more than design choices in the covariance function; the paper alludes to further work to be done here to better understand this correspondence.

[^1]: {% include citation.html key="damianou2013" %}
