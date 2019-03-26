---
layout: post
title: "Forward vs Reverse KL Divergences"
date: 2019-03-25
mathjax: true
---

# High Level Overview

TBC...

# Background

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

What is often skirted over is why don't we use the forward divergence? This would look something like this:

$$
\begin{equation}
D_{\text{KL}}\left(P(z|x) || Q(z) \right) = \mathbb{E}_{z \sim P} \left[ \log \frac{P(z|x)}{Q(z)} \right].
\label{eq:ForKL}
\end{equation}
$$

Clearly these are different quantities (KL divergence is assymetric after all), and therefore optimising each one will result in different optimal approximate $$Q$$ distributions. We will therefore address the following:

* What does optimal look like in the reverse and forward divergences?
* Why does VI strictly use reverse KL divergences?
* How can we use forward divergences?

# Comparing Forward and Reverse KL Divergences

TBC...