---
layout: post
title: "Notes on: A Unified View of Piecewise Linear Neural Network Verification"
date: 2018-12-06
mathjax: true
---

Original Paper by: Rudy Bunel, Ilker Turkaslan, Philip H.S. Torr, Pushmeet Kohli, M. Pawan Kumar

# High Level Overview

1. This papers introduces a unifying framework based on Branch-and-Bound to address NN verification techniques based on Satisfiability Modulo Theory paradigms, such as Relplex and Planet
2. In doing so, it is easier to identify which parts of the existing literature can be improved upon, and thus it is easier to 'swap in' components from other areas of optimisation to deliver superlative performance (such as research in robust optimisation)
3. An additional dataset (PCAMNIST) is introduced to determine the effect of various architectural/property bound choices on performance

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

The paper motivates the unification of verification formalisms by taking the approach of proof by contradiction, whereby we try to find a counter-example that would falsify the property $P(\cdot)$. This requires the definition of variables corresponding to the inputs, hidden units, and outputs of the network, and then a set of contraints that such a counter-example would satisfy.

With this in mind, the paper frames all NN verification problems as the following: "a global optimization problem where the decision will be obtained by checking the sign of the minimum". The paper illustrates this with the following example:

$$
\begin{equation}
P(\hat{\textbf{x}}_n) \triangleq \textbf{c}^\top \hat{\mathbf{x}}_n > b
\end{equation}
$$

This is equivalent to adding an additional FC layer to the network with one output. If we then set the bias on this layer to $-b$, we can re-frame the problem as above, namely, if the global minimum is positive, then the property must be satisfied ($\textbf{c}^\top \hat{\mathbf{x}}_n > b$) and therefore True. Conversely, if the global minimum is negative, then the property must be False, and the minimiser obtains these counter-examples. In summary, we reduce the problem to that of finding whether the scalar output of a NN reaches a negative value. This gives rise to the following:

$$
\begin{equation}
\mathbf{l}_0 \leq \mathbf{x}_0 \leq \mathbf{u}_0
\label{eq:input}
\end{equation}
$$

$$
\begin{equation}
\hat{x}_n \leq 0
\label{eq:output}
\end{equation}
$$

$$
\begin{equation}
\hat{\mathbf{x}}_{i+1} = \mathbf{W}_{i+1}\mathbf{x}_i + b_{i+1} \qquad \forall i \in \{0,n-1\}
\label{eq:layers}
\end{equation}
$$

$$
\begin{equation}
\mathbf{x}_i = \max (\hat{\mathbf{x}_i},0) \qquad \forall i \in \{0,n-1\}
\label{eq:relu}
\end{equation}
$$

These represent the bounds on the input (Eq \ref{eq:input}), the bound on the output (Eq \ref{eq:output}, to satisfy the property), the neural network layers and parameters (Eq \ref{eq:layers}), and ReLU activation functions respectively (Eq \ref{eq:relu}).

Therefore we are required to find an assignment to all the above variables, and in doing so we find a counter-example. Conversely, to prove the property is satisfied, we must prove that no counter-examples exist.

The paper observes that the addition of the non-linearity in Eq \ref{eq:relu} transforms this from a Linear Programming problem to one that is NP-hard. Fortunately it is possible to convert this problem into a Mixed Integer Linear Programming (MIP) problem, which is still NP-hard, but there exists literature which are able to solve such problems relatively efficiently. To perform the conversion to a MIP problem, the paper uses big-M encoding:

$$
\begin{align}
x_i \leq 0 \quad \Longrightarrow \quad \boldsymbol{\delta}_i \in \{ 0, 1 \}^{h_i}, \qquad & \mathbf{x}_i \geq 0, \qquad \mathbf{x}_i \leq \mathbf{u}_i \cdot \boldsymbol{\delta}_i \\
& \mathbf{x}_i \geq \hat{\mathbf{x}}_i, \qquad \mathbf{x}_i \leq \hat{\mathbf{x}} - \mathbf{l}_i \cdot (1 - \boldsymbol{\delta}_i)
\end{align}
$$

where $h_i$ appears to be the number of nodes in the $i$-th layer (since we'd require a tuple with this many entries to describe the "state" of the ReLUs in that layer). We clearly observe that the $\boldsymbol{\delta}$ variable allows us to switch between the two modes of the ReLU (i.e., linear region or 0 region). The paper points out that due to the feed-foward structure of NNs, such bounds (starting at the input) can be propagated through the network one layer at a time.

# Brand-and-Bound for Verification

As soon as minimum-searching is mentioned, it is easy to consider the traditional approach of SGD. However such SGD-based methods are not feasible for verification purposes as there are no guarantees over global minima, hence we cannot guarantee we've not missed a counter-example. Consequently, another approach must be taken, which is where branch-and-bound paradigms are introduced. The paper makes the connection that in fact many of the existing verification algorithms are in fact based on branch-and-bound, specifically those introduced as Satisfiability Modulo Theories. This generalisation is shown below (taken from the paper):

<p align="center" >
<img src="/assets/img/algorithm_bab.png" alt="branch and bound algorithm" height="500"/>
</p>

What is important is to observe the highlighted areas of the algorithm, as these represent the areas where various BaB-based algorithms differ, namely:
* Search Strategy (`pick_out`): This selects the next domain to branch on.
* Branching Rule (`split`): Takes a domain, and splits it into various subdomains over which the bounds are computed.
* Bounding Methods (`compute_UB/LB`): Given an input domain, discover the upper and lower bound of the network.

General points raised in the paper include:
* Domains whose lower bound is higher than the current global upper bound can be pruned away, as these domains can't contain the global minimum
* Domains whose lower bound is greater than 0 can be pruned, as no counter-examples can exist in these
* Conversely, if a domain has an upper bound of less than 0, the algorithm can be stopped as a counter-example must exist
* Consequently, calculated lower bounds over domains should be as high as possible (and upper bounds as low as possible), as this allows for domain pruning (or early termination)

The paper then covers how two existing SMT-based verification approaches, Reluplex and Planet, can be generalised as a BaB method. The following points are made:
* The **search strategy** doesn't appear to prioritise domains
* The **branching rule** that splits the domain of the $j$-th neuron in the $i$-th into the linear and 0 regions ($$\{ x_{i[j]} = 0, \hat{x}_{i[j]} \leq 0 \}$$, $$\{ x_{i[j]} = \hat{x}_{i[j]} , \hat{x}_{i[j]} \geq 0 \} $$) 
* The **bounding method** is a convex relaxation; Planet introduces a tighter relaxation than that of Reluplex

Having framed the problem as a BaB paradigm, the paper offers ways that existing procedures can be improved, specifically:
* Having tighter bounds by either refining bound estimates after splitting, or by exploring higher orders of the Sherali-Adams hierarchy.
* Improving branching by splitting the domain on its input features instead. BaB-input splits on the longest dimension, whilst BaBSB splits along the dimension with the highest upper bound given a fast bound over each subdomain


# Experimental Setup/Analysis

A baseline is introduced, whereby Eqs \ref{eq:input}-\ref{eq:relu} are put into a standard optimizer (Gurobi), which therefore doesn't exploit the structure of the problem itself. The following datasets are used:
* CollisionDetection: Whether 2 vehicles will collide based on parameterised trajectories. There are 500 properties to test concerning the regions around the data in which the same classification is given.
* ACAS: In which 1 of 5 manouevers are recommended by a NN; 188 properties are to be verified concerning 10 macro-properties (i.e., $$\phi_2$$ states that if an intruder aircraft is distant and slow, the score of a COC (Clear-of-Conflict) should not be maximal).
* PCAMNIST: In which the MNIST dataset is transformed using PCA into various lower dimensions, and different architectures/input dimensions are experimented (to test scalability); digits are classified as either odd or even. The property to be tested is whether the score assigned to an odd class is greater than the even class plus some confidence. This dataset allows for extremely flexible configurations that test for the effect of individual changes in the test setup (margin size, hidden size, layers, etc.).

Observing the cactus graphs, the following can be said:
* On the shallower networks in the CollisionDetection task, the Planet-based schemes perform best, verifying almost all properties within 10s. Furthermore, all schemes appear to be able to verify all properties eventually.
* On the denser networks in the ACAS task, the Planet-based schemes perform poorly, verifying less than 50% of all properties before timing out. In this case the BaB-based schemes perform best, particularly the smart branching approach, which is two order of magnitudes better than Reluplex.
* On the PCAMNIST cactus plots, again BaBSB performs best. In terms of scaling, it is clear that larger networks (both depth/width/input dimension) take longer to test, and furthermore, tighter margins on the property take longer to prove.

# Conclusion

By unifying the NN verification framework, it is easier to identify which parts within existing algorithms can be further optimised to deliver improved performance, which is demonstrated by the relatively superlative performance of the BaBSB algorithm, which leverages techniques derived from robust optimisation.

[^1]: Wikipedia: Formal verification
[^2]: {% include citation.html key="berrada2018" %}
