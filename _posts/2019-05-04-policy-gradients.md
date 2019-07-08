---
layout: post
title: "Policy Gradient Methods"
date: 2019-05-04
mathjax: true
published: false
---

## Table of Contents

1. [Introduction](#introduction)
2. [Motivation](#motivation)
3. [Policy Gradient Theorem](#policy-gradient-theorem)
    1. [Finite Time Horizon](#finite-time-horizon)
    2. [Infinite Time Horizon with Discounting](#infinite-time-horison-with-discounting)
4. [Applications](#applications)
    1. [All-actions](#all-actions)
    2. [REINFORCE](#reinforce)
    3. [REINFORCE with Baseline](#reinforce-with-baseline)
5. [Actor-Critic](#actor-critic)
    1. [Derivation](#derivation)
    2. [Types](#types)
        1. [A2C](#a2c)
        2. [Trust Region](#trust-region)
        3. [Deterministic Policies](#deterministic-policies)

## Introduction

Policy gradient methods are an alternative approach to solving reinforcement problems to the standard action-value methods (i.e., SARSA, Q-learning). Instead of inferring a policy indirectly by choosing actions which maximise expected return (i.e., $$\operatorname*{argmax}_a Q(s,a)$$), we instead directly learn the actions conditioned on the state (i.e., $$\pi(a\vert s)$$).

The purest form of policy gradient methods leverage only the policy gradient theorem to learn the optimal policy, but we can actually combine policy gradients with action-value methods to create actor-critic methods, which represent state of the art in a variety of tasks[^1].

We will discuss the advantages of policy gradient approaches, how to derive them, and extensions of this framework including actor-critic models.

## Motivation

Policy gradients have some advantages over action-value methods. In no particular order:

* They can learn stochastic policies, which may be ideal in a given environment (such as poker, where randomness is required to play optimally).
    * This also means that stochastic policies naturally explore, but note that in practice some additional exploration is still required as they can fall into local minima otherwise.
* They can be parameterised in any way required, and this is a way to impart domain knowledge. For example, we can use them to parameterise a Gaussian with mean and variance, resulting in policies over continuous action spaces (not possible with action-value methods).
* They learn faster in the case of environments where the policy function is easier to approxiate than the action-value function, such as Tetris.

## Policy Gradient Theorem

Here we make concrete about how to directly determine an optimal (parameterised) policy from data. What is commonly looked over is that there are multiple ways of writing down the expression we wish to optimise. Three which are common are as follows:

* Average episode return: $$J(\theta) = v_{\pi_\theta}(s_0)$$
* Average value: $$J(\theta) = \sum_s \mu_{\pi_\theta}(s) v_{\pi_\theta}(s)$$
* Average reward per time step: $$J(\theta) = \sum_s \mu_{\pi_\theta}(s) \sum_a \pi(a\vert s)r(s,a)$$

where $$\mu_{\pi_\theta}(s)$$ represents the stationary distribution of a policy $$\pi_\theta$$. In simple terms, it tells us what proportion is spent in a state $$s$$ in the limit of $$t \rightarrow \infty$$ (more on this later).

It is worth noting here is that the first expression only applies for episodic environments, where episodes have definite termination/endings. The latter two approaches can be applied to continuing cases, with potentially infinite horizons (i.e., they may never terminate). Interestingly, it can be shown that if there exists both an asymptotic total average performance AND future discounted performance, the latter two expressions can be equivalent[^2]. For the purposes of this post we will concentrate on the average value for our infinite horison derviation, as this derivation is the standard approach seen in textbooks.

### Finite Time Horizon

To derive the policy gradient theorem for finite time horizon scenarios, we will turn to the average episode return:

$$
\begin{align}
J(\theta) = v_{\pi_\theta}(s_0) \label{eq:averageepisodereturn}
\end{align}
$$

As long as we can differentiate this expression, then we can turn to gradient descent (or ascent in this case) to find an (locally) optimal setting of $$\theta$$ to maximise $$J$$.

To determine a differentiable form, we can rewrite $$J(\theta)$$ as follows:


### Infinite Time Horizon

[^1]: {% include citation.html key="burda2018" %}
[^2]: {% include citation.html key="hutter2006" %}