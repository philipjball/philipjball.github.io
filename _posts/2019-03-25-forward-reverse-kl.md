---
layout: post
title: "Forward vs Reverse KL Divergences"
date: 2019-03-25
mathjax: true
---

# Introduction

In variational inference, we rely on the idea of turning a posterior Bayesian inference problem into one that involves the minimisation of a divergence (usually KL) between a simplifying variational distribution $$Q$$ and some true (intractable) posterior $$P$$. This is done to avoid the oftentimes intractable integrals that are required in the 'true' Bayesian form: 

$$
\begin{equation}
P(z|X) = \frac{P(X|z)P(z)}{\int P(z,X) \text{d}z}.
\label{eq:BayesPosterior}
\end{equation}
$$

Note the difficulty of integrating over the ENTIRE latent space.

In VI we rely on the 'reverse' KL divergence; assuming we wish to infer some posterior distribution over latent variables $$z$$:

$$
\begin{equation}
D_{\text{KL}}\left(Q(z)||P(z|x)\right) = \mathbb{E}_{z \sim Q} \left[ \log \frac{Q(z)}{P(z|x)} \right].
\label{eq:RevKL}
\end{equation}
$$

What is often skirted over is why don't we use the forward divergence? This is as follows:

$$
\begin{equation}
D_{\text{KL}}\left(P(z|x) || Q(z) \right) = \mathbb{E}_{z \sim P} \left[ \log \frac{P(z|x)}{Q(z)} \right].
\label{eq:ForKL}
\end{equation}
$$

Clearly these are different quantities (KL divergence is assymetric after all), and therefore optimising each one will result in different optimal approximate $$Q$$ distributions.

# Comparing Forward and Reverse KL Divergences

## Reverse KL

Let us consider the reverse KL divergence, recalling we wish to **MINIMISE** this quantity; as before:

$$
\begin{align}
D_{\text{KL}}\left(Q(z)||P(z|x)\right) &= \mathbb{E}_{z \sim Q} \left[ \log \frac{Q(z)}{P(z|x)} \right]
\end{align}
$$

There are two scenarios to consider:

1. When is this quantity large?
2. When is this quantity small?

Consider the fraction $$\frac{Q(z)}{P(z\vert x)}$$. This fraction becomes large when $$Q$$ is large, and/or $$P$$ is small. Conversely, this fraction becomes small when $$Q$$ is small, and/or when $$P$$ is large. The second point is interesting; consider a case where $$Q$$ is small AND $$P$$ is large. 

What about the $$Q$$ outside the logarithm? Given that $$Q$$ must distribute mass somewhere (and therefore must be 'large' somewhere, amplifying the value of $$\log \frac{Q(z)}{P(z\vert x)}$$), as long as that happens when $$P$$ is large we can minimise KL, and therefore it doesn't matter if parts of where $$P$$ is large are missed. This implies that $$Q$$ is able to effectively ignore parts of the true distribution which are probable, as long as it covers a part of $$P$$ which is probable.

## Forward KL

Let us consider the forward KL divergence, recalling we wish to **MINIMISE** this quantity; as before:

$$
\begin{align}
D_{\text{KL}}\left(P(z|x)||Q(z)\right) &= \mathbb{E}_{z \sim P} \left[ \log \frac{P(z|x)}{Q(z)} \right]
\end{align}
$$

Let's consider the same two questions:

1. When is this quantity large?
2. When is this quantity small?

Consider the fraction $$\frac{P(z\vert x)}{Q(z)}$$. This fraction becomes large when $$Q$$ is small, and/or $$P$$ is large. Conversely, this fraction becomes small when $$Q$$ is large, and/or $$P$$ is small.

Turning to the $$P$$ outside of the logarithm, any distribution will have regions of higher probability, therefore we must ensure the $$\frac{P(z\vert x)}{Q(z)}$$ is small where this is this case. Consider a simple case of a bi-modal $$P$$, and uni-modal $$Q$$; here there will be two regions of high probability we must approximate with $$Q$$. We need to ensure therefore $$Q$$ is large in *BOTH* these regions, since $$P$$ will be large (see reasoning above). Therefore the only way we can achieve this is to spread the mass of $$Q$$ over both these modes.

## An Illustration

We have theorised the following behaviour for reverse and forward KL divergence minimisation:

1. In reverse KL, the approximate distribution $$Q$$ will distribute mass over a mode of $$P$$, but not all modes (mode-seeking)
2. In forward KL, the approximate distribution $$Q$$ will distribute mass over all modes of $$P$$ (mean-seaking)

To verify this, we can run the following experiment: given a multi-modal $$P$$ and a uni-modal $$Q$$, will $$Q$$ be 'mode-seeking' or 'mean-seeking' for each divergence minimisation? For illustrative purposes, consider the following 'true' bi-modal Gaussian:

<p align="center" >
<img src="/assets/img/P_true.svg" alt="True Distribution P(x)" height="300"/>
</p>

To fit the hypothetical $$Q$$ distributions, we approximate the KL divergence using Monte Carlo integration:

$$
\begin{align}
D_{\text{KL}}\left(A(x)||B(x)\right) &= \mathbb{E}_{x \sim A} \left[ \log \frac{A(x)}{B(x)} \right]\\
&\approx \frac{1}{n} \sum^n_{i=1} \left[ \log(A(x_i)) - \log(B(x_i)) \right] \label{eq:MCapprox}
\end{align}
$$

where in the second line, as $$n \rightarrow \infty$$ we estimate the true quantity, and all $$x_i$$ are drawn from the distribution $$A$$. Eq \ref{eq:MCapprox} allows us to calculate the KL divergence given samples, which we can easily draw the Gaussian models shown here. Therefore we run a grid search over the parameters of $$\mu$$ and $$\sigma$$ for $$Q$$ (N.B.: taking derivatives, it can be shown trivially that the optimal parameters for the forward KL is just the mean and variance of the samples from $$P$$, driving home the mean-seeking behaviour) and determine which configuration gives the lowest KL divergence in the case of reverse and forward approaches. Doing so reveals the following:

<p align="center" >
<img src="/assets/img/KL_compare.svg" alt="Distribution Comparison" height="300"/>
</p>

We have therefore shown that the hypothesised behaviour of the two divergence approaches is in fact correct.