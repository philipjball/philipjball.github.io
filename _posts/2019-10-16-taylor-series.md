---
layout: post
title: "A Taylor-Derived Series"
date: 2019-10-16
mathjax: true
---

When reading the [Evolutionary Strategies paper](https://arxiv.org/abs/1703.03864), I came across a [proof](https://davidbarber.github.io/blog/2017/04/03/variational-optimisation/) for the smoothed objective function that instead utilised a Taylor series approximation which initially confused me. I'm familiar with the idea of of finding a local approximation to a function at some point, but the series introduced here kind of flips that on its head; instead a local fixed point itself is the variable, and we're interested in the small perturbations around this point. I'd argue this treatment isn't technically the Taylor series, but is a series derived from it. To illustrate the point, consider the following standard Taylor series expansion:

$$
\begin{align}
F(x) = F(a) + F'(a)(x-a) + \frac{F''(a)}{2!}(x-a)^2 + \frac{F'''(a)}{3!}(x-a)^3 + \dots
\end{align}
$$

So we're basically saying at some point $$a$$, what is some *local* approximation to this function? (Thus implicitely $$x-a$$ should be small for lower powers).

An alternative way of viewing this is that we actually just care about a single point, say $$x_0$$, and want to understand how this is affected by small perturbations. In this case we can rewrite things as follows:

* $a = x_0$
* $x-a = \epsilon$

Substituting this into the original series we get the neat result (note how we recover $$x$$ by $$x_0 + \epsilon = x$$):

$$
\begin{align}
F(x_0 + \epsilon) = F(x_0) + F'(x_0)\epsilon + \frac{F''(x_0)}{2!}\epsilon^2 + \frac{F'''(x_0)}{3!}\epsilon^3 + \dots
\end{align}
$$

This formulation for me really drives home the fact that the Taylor series, for lower powers anyway, is a local approximation. To understand why we'd want to use this form it might be helpful to simply substitute the local point of interest $$x_0$$ with $$x$$, and do the following, assuming the perturbations are normally distributed:

$$
\begin{align}
F(x + \epsilon) = F(x) + F'(x)\epsilon + \frac{F''(x)}{2!}\epsilon^2 + \dots\\
\epsilon F(x + \epsilon) = \epsilon F(x) + \epsilon^2 F'(x)\epsilon + \epsilon^3 \frac{F''(x)}{2!} + \dots\\
\mathbb{E}[\epsilon F(x + \epsilon)] = 0 + \sigma^2 F'(x) + 0 + ... \\
F'(x) \approx \frac{1}{\sigma^2}\mathbb{E}[\epsilon F(x + \epsilon)]
\end{align}
$$

As we can see this is a neat way to approximate the gradient through samples, and indeed is how it is done in the ES paper (and many others).

There is in fact an alternative way to achieve the same gradient approximation by making Gaussian assumptions over the distribution of parameters and then using the log-derivative trick (which is the approach they take in the paper), which I may cover in a later post.