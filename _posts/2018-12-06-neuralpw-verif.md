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

Wikipedia[^1] defines formal verification as "the act of proving or disproving the correctness of intended algorithms underlying a system with respect to a certain formal specification or property, using formal methods of mathematics". In light of adversarial attacks[^2], and the increased ability of neural networks to function in safety-critical tasks (such as driving) the idea of safety in deep-learning is more relevant than ever. Formal verification helps us achieve this by  

# Problem Specification



# Verification Formalism



# Brand-and-Bound for Verification



# Experimental Setup



# Analysis



# Conclusion


[^1]: 
    Wikipedia: [Formal verification](https://en.wikipedia.org/wiki/Formal_verification)

[^2]: 
    {% include citation.html key="berrada2018" %}