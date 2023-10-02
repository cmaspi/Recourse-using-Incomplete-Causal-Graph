---
marp: true
math: mathjax
paginate: true
---

# Algorithmic Recourse Under Incomplete Causal Graph

---
 
Algorithmic Recourse: from Counterfactual Explanations to Interventions
===
 
## ACM 2021

---
Algorithmic Recourse
===
The systematic process of reversing unfavourable decisions by algorithms and bureaucracies across a range of counterfactual scenarios.
 
---
 
Problem Formulation
===

$$x^{\text{*CFE}} \in \underset{x}{argmin}\quad dist(x, x^F) \quad s.t. \; h(x) \neq h(x^F), x\in \mathcal{P}$$
 
---
Contd.
===
 
## Ustun et al.
$$
\begin{align}
\delta^* \in \underset{\delta}{argmin} \quad cost(\delta; x^F) \quad s.t.\; & h(x^{\text{CFE}}) \neq h(x^\text{F}),\\
& x^{\text{CFE}} = x^{\text{F}} + \delta,\\
& x^{\text{CFE}} \in \mathcal{P},\, \delta \in \mathcal{F} 
\end{align}
$$
 
---
Why does the above formulation fail?
===
- Example: Consider an example where an individual has an annual salary of \$75,000 and an account balance \$25,000. Say the model is 
$$h = sgn(X_1+5X_2 - \$225,000)$$
- The counterfactual explanation could be 
    1. Annual salary: \$100,000 or
    2. Bank Balance: $30,000
 
In a world where home-seekers save %30 of their salary, a salary increase of 14% would result in positive decision of loan-granting algorithm.
 
---
Structural Causal Model
===

<style>
    img[alt~="center"]{
        display: block;
        margin: 0 auto;
    }
</style>

![width:800px center](./figs/SCM_ex1.png)

---

# Proposition-1
A CFE-based action, $\mathbf{A}^{CFE}$, where $I=\{i|\delta_i^*\neq 0\}$, performed by individual $x^F$, in general results in the structural counterfactual, $x^{SCF}=x^{*CFE}:=x^{F}+\delta^*$, if and only if, the set of descendants of the acted upon variables, determined by $I$, is the empty set.

# Corollary-1
If the true world $\mathcal{M}$ is independent, i.e. all the observed features are root-nodes, then CFE-based actions always guarantee recourse.

---
Causal Persepective
===
$$
\begin{align}
\mathbf{A}^* \in \underset{A}{argmin}\quad &cost(\mathbf{A}; x^F)\\
s.t.\quad & h(x^{\text{SCF}}) \neq h(x^\text{F}),\\
& x^{\text{CFE}} = \mathbb{F_\mathbf{A}(\mathbb{F}^{-1}(x^\text{F}))}
& x^{\text{CFE}} \in \mathcal{P},\, \mathbf{A} \in \mathcal{F} 
\end{align}
$$

---

# Proposition-2
Given an individual $x^F$ observed in world $\mathcal{M}$, a family of feasible actions $\mathcal{F}$. Assume that there exists CFE-based actions $A^{CFE}\in \mathcal{F}$ that achieves recourse, i.e., $h(x^F)\neq h(x^{*CFE})$. Then, $cost(A^*;x^F) \leq cost(A^{CFE}; x^F)$

---

# Algorithm
1) **Abduction:** uniquely determines the values of all exogenous variables.
2) **Action:** modify the SCM according to the hypothetical interventions. ($\mathbb{F}_A$)
3) **Prediction:** Determine the values of of all endogenous variables.

The assignment of structural counterfactual values can generally be written as

$$
\begin{align}
x_i^{\text{SCF}} =& [i\in I]\cdot (x_i^\text{F}+\delta_i)\\
&+[i\notin I]\cdot (x_i^\text{F}+f_i(\textbf{pa}_i^\text{SCF})+f_i(\textbf{pa}_i^\text{F}))
\end{align}
$$

---
# Limitations
1) The underlying causal model is rarely known in practice.
2) The assumption of one-to-one mapping from real world actions to interventions on endogenous variables may not hold.

---
# Limitations

<span style="color:green;">1) The underlying causal model is rarely known in practice.</span>
2) The assumption of one-to-one mapping from real world actions to interventions on endogenous variables may not hold.

---
Algorithmic Recourse under imperfect causal knowledge: A probabilistic approach
===

## NeurIPS 2020
---

# Setting
The causal graph is known but the structural equations are not known.

---
# No Recourse Guarantees for unknown structural equations
**Example:** *Consider the following two SCMs $\mathcal{M_A}$ and $\mathcal{M_B}$.* 
![center](figs/no_recourse.png)

---
*Choose $U_1, U_2 \sim Bernoulli(0.5)$ and $U_3 \sim Uniform(\{0,\dots,K\})$ independently in both $\mathcal{M_A}$ and $\mathcal{M_B}$ with structural equations*
![Alt text](figs/equations.png)

*Then $\mathcal{M_A}$ and $\mathcal{M_B}$ both imply exactly the same observational and interventional distributions, and thus indistiguishable from empirical data. However, having observed $x^\text{F}=(1,0,0)$, they predict differebt counterfactuals had $X_1$ been 0, i.e., $x^\text{SCF}(X_1=0)=(0,0,0)$ and $(0,0,K)$, respectively.*

---

# Proposition
Unless the set of descendants of interevend-upon variables in empty, algorithmic recourse can in general, be guaranteed only if the true structural equations are known, irrespective of the amount and type of available data.

---

Method
---
The learned causal model could be imperfect due to finite sample of the observed data, or due to model misspecification.

The authors adopt a Bayesian approach to account for the uncertainty in the estimation of the structural equations. They assume additive gaussian noise and rely on probabilistic regression using a Gaussian process (GP) prior over the functions $f_r$.

---
# GP-SCM
$$
\begin{align}
X_r:&=f_r(\mathbf{X}_{\text{pa}(r)})+U_r\\
where \quad & f_r \sim \mathcal{GP}(0, k_r),\quad U_r\sim\mathcal{N}(0, \sigma_r^2), \quad r\in [d]
\end{align}
$$

---

Add more math (proposition 5, 6 from the paper)

---

Algorithm for optimization

---
# Limitations




---

# Learning Representation for counterfactual inference


---

# Assumptions
1. Acyclicity: Causal structure can be represented by DAG
2. Markov Property: all nodes are independent of their non-descendants when conditioned on their parents
3. Faithfulness: All conditioned independences in true underlying distribution p are represented in G
4. Sufficiency: any pair of nodes in G has no common external cause.

---

# Tricks for Causal Discovery

Trick | Algorithm
 --- | --- 
Conditional Independence Testing | PC, Fast Causal Inference (FCI), Inductive Causation (IC)
Greedy Search of DAG Space | Greedy Equivalence Search (GES), Greedy Interventional Equivalence Search (GIES), Concave Penalized Coordinate Descent with Reparametrization (CCDr)
Exploiting Asymmetry | Linear Non-Gaussian Acyclic Model (LiNGAM), Non-linear Additive Noise Model, Post-non linear Causal Model (PNL), Granger Causality
Hybrid | Structural Agnostic Modeling (SAM), Causal Additive Model(CAM), Causal Generative Neural Network (CGNN)

---

# Trick-1: Conditional Independence Testing
<style>
    img[alt~="center"]{
        display: block;
        margin: 0 auto;
    }
</style>
The key idea is that two statistically independent variables are not causally linked.

![center width:600](figs/CIT.png)
1. (B) Start with fully connected graph
2. (C), (D) remove edges b/w statistically independent variables.
3. (E) Orient Colliders
4. (F) Additional Constraints.

---

# Trick-2: Greedy Equivalence Search (GES)

The Greedy Equivalence Search (GES) algorithm uses this trick. GES starts with an empty graph and iteratively adds directed edges to maximize the improvement in a model fitness measure (i.e. score). An example score is the Bayesian Information Criterion (BIC) 

---

# Trick-3: Exploiting Asymmetries
1. **Time Asymmetry:** Granger Causality (Cause precedes the effect)
2. **Complexity Asymmetry:** Simpler models are better.
3. **Functional Asymmetry:** Models that fit a relationship better, are better models.


