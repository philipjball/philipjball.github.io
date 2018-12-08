---
layout: post
title: "Notes on: A Unified View of Piecewise Linear Neural Network Verification"
date: 2018-12-06
mathjax: true
---

Original Paper by: Rudy Bunel, Ilker Turkaslan, Philip H.S. Torr, Pushmeet Kohli, M. Pawan Kumar

# High Level Overview

1. Unify the existing literature on neural network verification, namely as special cases of 'Branch-and-Bound' optimisation.
2. Propose a new set of benchmarks to compare existing (and future) verification algorithms.
3. Speeding up existing state-of-the-art methods by 2 orders of magnitude through algorithmic improvements.

# Introduction

Wikipedia[^1] defines formal verification as "the act of proving or disproving the correctness of intended algorithms underlying a system with respect to a certain formal specification or property, using formal methods of mathematics". In light of adversarial attacks[^2], and the increased ability of neural networks to function in safety-critical tasks (such as driving) the idea of safety in deep-learning is more relevant than ever. Formal verification helps us achieve this by ensuring specific properties of the 'black-box' that is the deep neural-network.

# Problem Specification

In the paper we are given the following presentation of verification within a neural network $f({\mathbf{x}})$:

$$
\begin{equation}
\mathbf{x}_0 \in \mathcal{C}, \quad \hat{\mathbf{x}}_n = f(\mathbf{x}_0) \Longrightarrow P(\hat{\mathbf{x}}_n)
\end{equation}
$$

where $\mathcal{C}$ is the bounded input domain, $\hat{\mathbf{x}}$ is the network's output, and $P(\cdot)$ is some property we wish to verify. The paper motivates this with the example of a neural network's robustness to adversarial examples, given an input $\mathbf{a}$ with label $y_n$:

$$
\begin{equation}
\mathcal{C} \triangleq \{\mathbf{x}_0 | \: \| \mathbf{x}_0 - \mathbf{a} \|_\infty \leq \epsilon\}, \quad P(\hat{\mathbf{x}}_n)= \{ \forall y \quad \hat{x}_{n[y_a]} > \hat{x}_{n[y]} \}.
\end{equation}
$$

In other words, given a perturbation in the infinity-norm of at most $\epsilon$, ensure that we continue to classify this data point as $y_a$.

The paper concentrates on problems concerning piece-wise neural networks (PL-NNs). These are effectively neural networks that have (piece-wise) linear activation functions, such as ReLU. What this allows us to do is decompose the input space $\mathcal{C}$ into a set of polyhedra $\mathcal{C}_i$, and as a result the [restriction](https://en.wikipedia.org/wiki/Restriction_(mathematics)) of $f$ (i.e., $$f \mid_{\mathcal{C}_i}$$) to $\mathcal{C}_i$ is a linear function.

The paper considers the case of properties that are Boolean formulas over linear inequalities, as alluded to in the example above (i.e., ensuring the output of the original label is larger than the one of an other class given a noise perturbation).

The paper doesn't consider the following:
* Additional assumptions, such as double differentiability, limitation to binary activation values, single linear domain
* Methods without formal guarantees, such as limited perturbation sets, over-approximation methods

# Verification Formalism



# Brand-and-Bound for Verification



# Experimental Setup



# Analysis



# Conclusion


[^1]: 
    Wikipedia: [Formal verification](https://en.wikipedia.org/wiki/Formal_verification)

[^2]: 
    {% include citation.html key="berrada2018" %}
