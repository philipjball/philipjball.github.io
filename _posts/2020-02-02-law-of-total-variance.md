---
layout: post
title: "Law of total expectation and Law of total variance"
date: 2020-02-02
mathjax: true
---

A useful reminder for two laws for when we have conditional distriubtions (i.e., Rao-Blackwell).

Before we begin, a simple notational point about conditional expectations that tripped me up; conventially when we write a conditional expectation $\mathbb{E}\_X [X\vert Y]$ this is equivalent (discrete case) to $\sum_{x \in \mathcal{X}} xP(X=x\vert Y)$. I'm personally more familiar with the expectation notation that is seen in machine learning, and if we were to write in this style, I'd expect something like this: $\mathbb{E}_{x\sim P(x\vert Y)}[X=x]$[^1].

## Law of Total Expectation

The law states that

$$
\begin{align}\label{eq:total_expectation}
\mathbb{E}_X[X] = \mathbb{E}_Y[\mathbb{E}_X[X|Y]].
\end{align}
$$

I'm going to make some assumptions and be loose with notation[^2] (i.e., non-infinite absolute integrands, disregard some measure theory) and show a quick proof of this in the continuous case. Suffice to say the discrete case follows by basically replacing integrals with sums.

$$
\begin{align}
\mathbb{E}_Y[\mathbb{E}_X[X|Y]] &= \int p(y) \int x p(x|y) \mathrm{d}x \mathrm{d}y \\
                                &= \int p(y) \int x \frac{p(x,y)}{p(y)} \mathrm{d}x \mathrm{d} y \\
                                &= \int \frac{p(y)}{p(y)} \int x p(x,y) \mathrm{d}x \mathrm{d} y \\
                                &= \int \int x p(x,y) \mathrm{d}y \mathrm{d} x \\
                                &= \int x \int p(x,y) \mathrm{d}y \mathrm{d} x \\
                                &= \int x p(x) \mathrm{d}x \\
                                &= \mathbb{E}_X[X].
\end{align}
$$

Thus 'proving' the original Equation \ref{eq:total_expectation}.

## Law of Total Variance

This follows on nicely from previous law because its proof relies on it. The law states that

$$
\begin{align}
\mathbb{V}_X[X] = \mathbb{E}_Y[\mathbb{V}_X[X|Y]] + \mathbb{V}_Y[\mathbb{E}_X[X|Y]]
\end{align}
$$

where $\mathbb{V}_\cdot[\cdot]$ is the variance of a variable.

This can be shown as follows:

$$
\begin{align}
\mathbb{V}_X[X] &= \mathbb{E}_X[X^2] - \mathbb{E}_X[X]^2 \\
                &= \mathbb{E}_Y[\mathbb{E}_X[X^2|Y]] - \mathbb{E}_Y[\mathbb{E}_X[X|Y]]^2 \\
                &= \mathbb{E}_Y[\mathbb{V}_X[X|Y] + \mathbb{E}_X[X|Y]^2] - - \mathbb{E}_Y[\mathbb{E}_X[X|Y]]^2 \\
                &= \mathbb{E}_Y[\mathbb{V}_X[X|Y]] + \left( \mathbb{E}_Y[\mathbb{E}[X|Y]^2] - \mathbb{E}_Y[\mathbb{E}_X[X|Y]]^2 \right) \\
                &= \mathbb{E}_Y[\mathbb{V}_X[X|Y]] + \mathbb{V}_Y[\mathbb{E}_X[X|Y]].
\end{align}
$$

This gives us two things:
1. A neat mathematical identity to use when we need to express the variance of a random variable as some other variable it may depend on.
2. Observing the two terms on the RHS, we can decompose the variance of any R.V. into two terms. Consider that we vary the 'treatment' r.v. $Y=y$. The first is the expectation of the conditional variance, which intuitively gives us the unexplained variance (i.e., within a given sample $Y=y$). The second is the variance of the conditional expectation, which intuitively gives us the explained variance (i.e., over all samples $Y=y$).

---

[^1]: An interesting intuition when working with conditional expectations. What is the taxonomy of $E[Y\vert X]$? Turns out it's a random variable; consider for a fixed $X$ that $E[Y\vert X=x] = f(x)$. Therefore it follows that $E[Y\vert X] = f(X)$, which must be a random variable. This explains why the outer expecation is w.r.t. $Y$ in the Law of Total Expectation; it inherits randomness from $Y$, since we integrate out $X$ by taking the expectation.
[^2]: Full disclosure, I studied engineering, not mathematics at undergraduate.