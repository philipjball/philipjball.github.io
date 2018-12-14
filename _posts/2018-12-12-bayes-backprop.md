---
layout: post
title: "Notes on: Weight Uncertainty in Neural Networks"
date: 2018-12-06
mathjax: true
---

Original Paper by: Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, Daan Wierstra

# High Level Overview

* This paper generalises the reparameterisation trick to the weights of a neural network, and furthermore allows for the use of non-Gaussian prior distributions
* The Bayesian paradigm naturally extends to the subsequent decisions made by such networks, offering uncertainty estimates over outputs
* Furthermore the uncertainty over decisions can be naturally extended to bandit/RL tasks, whereby exploration/exploitation arises naturally due to uncertainty in weights affecting uncertainty in the selected actions via Thompson sampling

# Introduction

This paper merges the areas of Bayesian machine learning with variational methods in the context of training neural networks. By doing this, it is then possible to leverage Bayesian concepts to aid with training, such as uncertainty estimates (which can be leveraged in reinforcement learning tasks requiring exploration), and regularisation via a compression cost, which can be described as the tradeoff between fitting the data well, and staying close to the prior distribution over the weights.

This paper introduces a distribution over the weights instead of single-valued parameters. This allows the network the freedom to express uncertainty in the data by adjusting (for example) its variance over certain weights. Furthermore, by employing a distribution of weights, the paper in effect trains an infinite ensemble of networks, since all that is needed to create a new member is to simply sample the weights. In fact it is this uncertainty in the weights that encodes regularisation, since it captures the uncertainty of which neural network would be appropriate to model certain data.

# Point Estimates of Neural Networks

The paper introduces the following notation to represent the probabilisitc model of the neural network:

$$
\begin{equation}
P(\mathbf{y}\vert\mathbf{x},\mathbf{w})
\end{equation}
$$

where $$\mathbf{x} \in \mathbb{R}^p$$ is the input, $$\mathbf{w}$$ are the parameters/weights, and $$\mathbf{y} \in \mathcal{Y}$$ is the output given the input. For classification, $$\mathcal{Y}$$ is a categorical distribution, and for regression, $$\mathcal{Y}$$ is $$\mathbb{R}$$, and $$P(\mathbf{y}\vert\mathbf{x},\mathbf{w})$$ is a Gaussian distribution, which implies a squared loss (recall the term in the exponent and the nautral logarithm). When training traditional neural networks, we can view this as maximum likelihood in the following form given $$n$$ datapoints:

$$
\begin{align}
\mathbf{w}^{\text{MLE}} &= \arg\max_{\mathbf{w}} \log P(\mathcal{D} \vert \mathbf{W}) \\
& = \arg\max_{\mathbf{w}} \log \Pi^{n}_{i} P(\mathbf{y}_i \vert \mathbf{x}_i, \mathbf{w})\\
& = \arg\max_{\mathbf{w}} \Sigma^{n}_{i} \log P(\mathbf{y}_i \vert \mathbf{x}_i, \mathbf{w})
\end{align}
$$

which is calculated using gradient methods, such as SGD. It is also possible to compute a posterior setting of the weights *given* the data using maximum a posteriori inference (MAP):

$$
\begin{align}
P(\mathbf{w} | \mathcal{D}) &= \frac{P(\mathbf{w})P(\mathcal{D} | \mathbf{w})}{P(\mathcal{D})} \label{eq:bayes}\\
&\propto P(\mathbf{w})P(\mathcal{D} | \mathbf{w}).
\end{align}
$$

Therefore,

$$
\begin{align}
\mathbf{w}^{\text{MAP}} &= \arg\max_{\mathbf{w}} \log P(\mathbf{w} \vert \mathcal{D}) \\
& = \arg\max_{\mathbf{w}} \log P(\mathbf{w})P(\mathcal{D} | \mathbf{w})\\
& = \arg\max_{\mathbf{w}} \log P(\mathbf{w}) + \mathbf\Sigma^{n}_{i} \log P(\mathbf{y}_i \vert \mathbf{x}_i, \mathbf{w}).
\end{align}
$$

We observe that when the prior distributon over the weights is Gaussian, we retrieve standard L2 regularisation, and when the prior over the weights is a Laplace distribution, we get L1 regularisation.

# Being Bayesian by Backpropagation

Whilst $$\mathbf{w}^{\text{MAP}}$$ is Bayesian in a sense (since it relies on updating a prior based on observed (training) data $$\mathcal{D}$$), it does not calculate the 'true' posterior distribution over the weights (i.e., $$P(\mathbf{w}\vert \mathcal{D})$$). This is hard to calculate analyatically since the denominator in Eq \ref{eq:bayes} is computationally difficult for large weight values, as we need to integrate over the ENTIRE weight space, which usually is very high-dimensional (x,000,000s):

$$
\begin{equation}
P(\mathcal{D}) = \int P(\mathbf{w} | \mathcal{D}) P(\mathbf{w}) d\mathbf{w}.
\end{equation}
$$

It is important that we are able to retrieve a true distribution $$P(\mathbf{w}\vert\mathcal{D})$$ since this allows us to determine the distribution over the network predictions given some test data:

$$
\begin{align}
P(\hat{\mathbf{y}}\vert \hat{\mathbf{x}}, \mathcal{D}) &= \int P(\mathbf{w} \vert \mathcal{D}) P(\hat{\mathbf{y}} | \hat{\mathbf{x}}, \mathbf{w}) d\mathbf{w}\\

&= \mathbb{E}_{P(\mathbf{w}\vert \mathcal{D})}[P(\hat{\mathbf{y}} | \hat{\mathbf{x}}, \mathbf{w})]
\end{align}
$$

It is possible to estimate this quantity using variational inference, which provides a lower bound on the true posterior distribution, bounded by the Kullback-Leibler (KL) divergence between the approximating distribution and the true distribution. Therefore when the approximation is good, we effectively retrieve the true distribution. Importantly, it is far easier to calculate this approximate posterior compared with the true posterior.

We can parameterise this approximating distribution by $$\theta$$, giving $$q(\mathbf{w}\vert \theta)$$. Calculating the minimum KL divergence between this and the true distribution yields the following:

$$
\begin{align}
\theta^* &= \arg\min_{\theta} D_{KL}[q(\mathbf{w}\vert \theta) \| P(\mathbf{w}\vert \mathcal{D}))] \\
&= \arg\min_{\theta} \int q(\mathbf{w}\vert\theta) \log \frac{q(\mathbf{w}\vert \theta)}{P(\mathbf{w})P(\mathcal{D}|\mathbf{w})} d\mathbf{w} \\
&= \arg\min_{\theta} \int \left[ q(\mathbf{w}\vert\theta) \log \frac{q(\mathbf{w}\vert \theta)}{P(\mathbf{w})} - q(\mathbf{w}\vert\theta) \log P(\mathcal{D}|\mathbf{w}) \right] d\mathbf{w} \\
&= \arg\min_{\theta} D_{KL}[q(\mathbf{w}\vert \theta) \| P(\mathbf{w})] - \mathbb{E}_{q(\mathbf{w}\vert \theta)}\left[ \log P(\mathcal{D}\vert \mathbf{w}) \right]. \label{eq:variational}
\end{align}
$$

We observe that Eq \ref{eq:variational} naturally gives rise to weight regularisation; the first KL term ensure that our learnt weights maintain the distributional characteristics of the prior, thus characteristics such as 'spiky', low entropy weight distributions (i.e., symptomatic of overfitting) are discouraged. The paper gives the following final form for this identity termed 'variational free energy':

$$
\begin{equation}
\mathcal{F}(\theta, \mathcal{D})= D_{KL}[q(\mathbf{w}\vert \theta) \| P(\mathbf{w})] - \mathbb{E}_{q(\mathbf{w}\vert \theta)}\left[ \log P(\mathcal{D}\vert \mathbf{w}) \right].\label{eq:freeenergy}
\end{equation}
$$

The paper now introduces a generalisation of the Reparameterisation Trick[^1], whereby a random variable can be written in terms of a 'simple' parameterised random variable given a transformation (i.e., $$\mathbf{w} = t(\theta, \epsilon)$$). This allows the derivative of an expectation to be written as the expectation of a derivative, thus the gradients are able to flow through the network so that parameters can be learnt. Assuming we can write $$q(\mathbf{w} \vert \theta)d\mathbf{w} = q(\epsilon) d\epsilon$$:

$$
\require{cancel}
\begin{align}
\frac{\partial}{\partial \theta} \mathbb{E}_{q(\mathbf{w}\vert \theta)} \left[ f(\mathbf{w},\theta) \right] &= \frac{\partial}{\partial \theta} \int f(\mathbf{w},\theta) q(\mathbf{w}\vert \theta) d\mathbf{w} \\
&= \frac{\partial}{\partial \theta} \int f(\mathbf{w},\theta) q(\epsilon) d\epsilon \\
&= \int \frac{\partial}{\partial \theta} f(\mathbf{w},\theta) q(\epsilon) d\epsilon \\
&= \int \left[ \frac{\partial}{\partial \theta}[f(\mathbf{w},\theta)]\cdot q(\epsilon) + \cancelto{0}{\frac{\partial}{\partial \theta}[q(\epsilon)]\cdot f(\mathbf{w},\theta)} \right] d\mathbf{w}\\
&= \mathbb{E}[ \frac{\partial f(\mathbf{w},\theta)}{\partial \theta} + \frac{\partial f(\mathbf{w},\theta)}{\partial \mathbf{w}} \frac{\partial \mathbf{w}}{\partial \theta} ]
\end{align}
$$

Observe that this does not depend on a random variable that is parameterised by a Gaussian, and can be extended to any distribution over the random variables we are reparamaterising. Of course if we were to substitute this into Eq \ref{eq:freeenergy}, we observe that the KL divergence may not be tractable without Gaussians or Binomials. Recalling the KL divergence is simply an expectation, we make the following approximation:

$$
\begin{equation}
\mathcal{F}(\mathcal{D},\theta) \approx \sum_{i=1}^n \log q(\mathbf{w}^{(i)}\vert \theta) - \log P(\mathbf{w}^{(i)}) - \log P(\mathcal{D} \vert \mathbf{w}^{(i)}) \label{eq:freeenergyapprox}
\end{equation}
$$

It turns out that this approximation through sampling is in fact a variance reduction technique known as 'common random numbers'. Interestingly, the paper notes that in cases with tractable KL divergences (i.e., Gaussians), using Eq \ref{eq:freeenergyapprox} in place of \ref{eq:freeenergy} actually had a negligble impact on final performance. This suggests this approximation does not have a detrimental effect on accuracy. Furthermore experimentally, hard to compute complexity costs performed better, hence the approximate form is important as it gives us more flexibility over the set of distributions.

In order to utilise the gradient updates to perform learning, the paper presents an example of a simple Gaussian posterior. In this case, the following steps are taken per step to update the learnt parameters, which in this case are the mean $$\mu$$ and standard deviation parameter $$\rho$$ such that $$\sigma = \log(1 + \exp(\rho))$$ so that the standard deviation is always positive.

1. Sample $$\epsilon \sim \mathcal{N}(0,I)$$
2. Transform using the parameterisation $$\mathbf{w} = \mu + \log(1 + \exp(\rho)) \circ \epsilon$$
3. Let $$f(\mathbf{w},\theta) = \log q(\mathbf{w} \vert \theta) - \log P(\mathbf{w}) - \log P(\mathcal{D} \vert \mathbf{w})$$ and $$\theta = (\mu, \rho)$$
4. Calculate gradients with respect to both parameters:
  * $\Delta_{\mu} = \frac{\partial f(\mathbf{w},\theta)}{\partial \mathbf{w}} + \frac{\partial f(\mathbf{w},\theta)}{\partial \mu}$
  * $\Delta_{\rho} = \frac{\partial f(\mathbf{w},\theta)}{\partial \mathbf{w}}\frac{\epsilon}{1 + \exp(-\rho)} + \frac{\partial f(\mathbf{w},\theta)}{\partial \rho}$
5. Perform an update:
  * $\mu \leftarrow \mu - \alpha \Delta_{\mu}$
  * $\rho \leftarrow \rho - \alpha \Delta_{\rho}$

The paper points out that conveniently, the shared $\frac{\partial f(\mathbf{w},\theta)}{\partial \mathbf{w}}$ term is simply the standard gradient update we get when learning neural networks.

As aforementioned, the paper generalises the reparameterization trick beyond that of Gaussians (despite the above example). Therefore to leverage this the additional flexibility over distributions, the paper introduces the scale mixture prior, which in effect is a mixture of Gaussians, each with 0 mean but different variances (i.e., one large, one small (relatively)). This prior is written as follows:

$$
\begin{equation}
P(\mathbf{w}) = \prod_j \pi \mathcal{N}(\mathbf{w}_j \vert 0, \sigma_1^2) + (1 - \pi) \mathcal{N}(\mathbf{w}_j \vert 0, \sigma_2^2).
\end{equation}
$$

This is similar to the spike-and-slab distributions presented previously in the literature. In this case however the prior parameters are shared amongst all the weights, and selected using cross-validation. Interestingly, using Type II maximum likelihood (a.k.a. emprical Bayes) methods to learn these from data actually gives worse results, since the paper determines the prior parameters learnt this way force the NN weights into early undesired local minima.

Finally, the paper addresses the incorporation of mini-batch based training into the variational approximations. Since the weights themselves are effectively drawn from a distribution, we simply treat the data the same, i.e., as a random sample from $\mathcal{D}$. This requires a way to weight the complexity term from the KL divergence with the likelihood term. One possible approach is:

$$
\begin{equation}
F_i^{EQ} = \frac{1}{M}D_{KL}[q(\mathbf{w}\vert\theta) \| P(\mathbf{w})] - \mathbb{E}_{q(\mathbf{w}\vert \theta)}[\log P(\mathcal{D}_i \vert \mathbf{w})].
\end{equation}
$$

We observe that summing over all the mini-batches retrieves the original Eq \ref{eq:freeenergy}. The approach used in the paper uses the following heuristic: early batches should have less influence on the weights due to the data, and as we observe more data, we ought to weight the data term more heavily. This is because the weights should have settled into a configuration that is representative of one that fits the data well. This is presented as follows:

$$
\begin{equation}
F_i^{EQ} = \pi_i D_{KL}[q(\mathbf{w}\vert\theta) \| P(\mathbf{w})] - \mathbb{E}_{q(\mathbf{w}\vert \theta)}[\log P(\mathcal{D}_i \vert \mathbf{w})]
\end{equation}
$$

where experimentally it is found that the following performs well: $\pi_i = \frac{2^{M-i}}{2^{M}-1}$, so that $\pi_1 \approx 0.5 $, and $\pi_M \approx 0$ when $M$ is large. Therefore earlier batches are dominated by the KL term, whilst later ones are dominated by the data term.

# Contextual Bandits

The paper makes an analogy between Thompson sampling and the drawing of random weights in the Bayesian neural network. Simply put, more confident weights will be drawn more often, thus representing the exploitation, whilst weights with less confidence will be drawn more randomly, representing exploration. As we train the NN, all the weights should decrease in variance, therefore we naturally perform the requisite exploration/exploitation requried to learn the task. This is in contrast to a non-Bayesian treatment, which usually involves setting an exploration variable $$\epsilon$$, and if necessary, annealing it as we learn more of the task. Explicitly, the paper outlines the following use of Thompson sampling in Bayesian NNs:

1. Draw weights from the variational/approximate posterior as $\mathbf{w} \sim q(\mathbf{w} \vert \theta)$
2. Receive a context $x$ (from the environment)
3. Pick the action $a$ greedily (i.e., such that it minimises regret, or maximimses expected return)
4. Receieve an actual reward $r$ from the environment
5. Update $\theta$ accordingly

The paper makes the observation that since initial weight distributions will be uniform, the resultant selected actions will also be uniformly distributed. This exploraiton is then naturally annealed as the weights gain confidence and uncertainty decreases. Interestengly, despite the literature suggesting variational approaches often under-estimate uncertainty, this does not appear to matter in the experiments run in this paper.

# Experiments

As with most papers, an MNIST benchmark is presented. In this case, Bayes-by-Backprop does best, with the scale-mixture prior performing significantly better than a standard Gaussian prior. It appears to outperform standard dropout methods, and doesn't seem to suffer from overfit given a large number of epochs (supposedly due to the regularisation), although the graphs don't really extend long enough to show this unequivocally.

Things to note are the distribution of the weights themselves; Bayes-by-Backprop appears to learn weights which are far more entropic and have a greater range than those learnt by either standard SGD or drop-out.

The paper also delves into the realm of neural network compression; namely deactivating some weights to 0 in the fully trained model to observe the decrease in performance. This is done by ordering the weights by their SNR, which is given by $\frac{\vert \mu_i \vert}{\sigma_i}$. What is interesting is that even when 95% of the lowest SNR weights are removed, *performance decreases by only 0.05%*. This feels counterintuitive since so many of the weights were non-zero, but perhaps this is due to the variance actually pushing the SNR of these weights down, hence their removal has negligible impact on final predictive power.

The paper also makes an interesting point concerning the storage of parameters at run-time; although we require more parameters to be generated overall (since we need mean AND variance), for techniques requiring ensembling (such as knowledge distillation), fewer parameters are stored. This is because we simply sample new networks in the ensemble, as opposed to have to store every network in the ensemble for test time.

As expected on regression tasks, in areas with little data (i.e., extrapolation) the Bayesian NN is able to correctly assign greater uncertainty to these regions.

Finally, a bandit/RL task is presented using a poisonous mushroom dataset, whereby the agent can either eat or skip a mushroom, receiving positive rewards for eating edible mushrooms, and negative rewards for eating poisonous ones. Again, the Bayesian NN leveraging Thompsons sampling performs best here.

# Discussion

Overall, Bayes-by-Backprop appears to merge all the positives of Bayesian approaches (i.e., uncertainty estimates, inherent regularisation) with those of neural networks (i.e., powerful predictivte models), and delivers performance comparable to/better than drop-out. Furthermore, its natural extension to RL tasks via Thompson sampling makes it a compelling candidate for application to such tasks.

[^1]: {% include citation.html key="kingma2013" %}
