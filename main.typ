#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node
#import "@preview/diverential:0.2.0": *

// double spacing
#set text(top-edge: 0.7em, bottom-edge: -0.3em, size: 11pt)
#set par(leading: 1em)

#set quote(block: true)
#show quote.where(block: true): it => {
  block(
    stroke: (left: 3pt + gray, rest: none),
    fill: luma(230),
    inset: (left: 4pt, rest: 10pt),
    above: 16pt,
    radius: 4pt,
    text(luma(80))[#it],
  )
}

// paragraph indents
#set par(first-line-indent: (
  amount: 1.5em,
  all: true,
))

// other math
#let bvec(x) = {
  $upright(bold(#x))$
}


= Transferability of PDE-discovery across physical regimes


In the early 1900s, a theory of solid states was taking form due to the discovery of the Drude model of metals (1900), X-ray crystallography (1913), and the discovery of superconductivity (1911). After WWII, the overarching physics grew larger in literature as new theories encompassed more material and phenomena including those of plasmas and liquids. So it quickly adopted a new name: condensed matter physics. Today, condensed matter physics has readily appreciable applications in materials science, quantum computing, and renewable energy technologies. It is often recognized as the most active subfield of physics by researcher count and impact.

Critically, recent computational breakthroughs have been advancing condensed matter physics rapidly. Particularly, CMP has been influenced by two main types of computational research: forward and inverse problems, the latter being far newer. The following review on literature and methods will be split as such, with a larger focus on inverse problems.

== Forward Problems

Luo et al. 2025 shows how methods of solving PDEs have evolved over time. "Classical" computational methods (i.e. non-ML or non-data-driven methods) include the finite element/volume/cell method (FEM/FVM/FCM). Most recently, machine learning is now being applied to solve PDEs efficiently. In particular, physics-informed machine learning is a hybrid approach, incorporating fundamental physical principles on top of data; this addition of principles enhances the model’s ability to generalize, especially in scenarios where data is scarce, noisy, or incomplete. Moreover, these methods are known to be very efficient for tackling both forward and reverse PDE problems (Raissi et al. 2019).


== Inverse Problems
// add potential visual

However, the discovery of governing equations from observational data represents a fundamental challenge whose literature has increased significantly over the past two decades. Specifically, data-driven methods via machine learning promise to automate this discovery process.

#quote[Given a known input $u_0$ and observed data $d$, what function $cal(F)_theta$ produced the data?]

#align(center)[
  #diagram(
    node-stroke: 1pt,
    edge-stroke: 1pt,
    mark-scale: 60%,
    $
      u_0 edge(->, cal(F)_theta) & d
    $,
  )
]

// please revise TODO: "initial data function" could also be "forcing function" right? cf. Steve Brunton FNOs
We will learn the function-to-function mapping, an _operator_, $cal(S)_theta : cal(U) -> cal(U)$ from an initial data function to solution function. For example, $u_0 mapsto S_t u_0 = u_t$, where $S_t$ might solve the differential equation

$ f(u_t, dv(u_t, t)) = 0 $

We can train a model to fit $cal(S)_theta$ given samples of $u_0, d$. However, instead of naively fitting the data, we can embed physical constraints into our model, be it soft constraints or direct restrictions via inductive biases. [] was first to realize that you can embed physics in the loss functional $J[hat(u)_t]$ of these $cal(S)_theta$-predicting architectures.

Inverse problems can be categorized further into

Papers that laid the foundation for solving inverse problems include .. . In 2007, Bongard and Lipson gave the first experiment of three methods for solving inverse problems. They introduced three major steps for symbolic regression of nonlinear systems. These are partitioning, automated probing, snipping.

=== Hadamard criteria

The Hadamard criterion for determining whether a particular inverse problem is "well-posed" requires solution existence, uniqueness, and stability; otherwise, it is considered "ill-posed". Inverse problems often fail the third condition, stability, as the behavior of several systems is very sensitive to various parameters; for inverse problems, this indicates that small perturbations in the solution operator (inverse operator) can lead to arbitrarily large variations in the inferred system parameters. To visualize this, consider we collect data $D$, and we are trying to recover the solution operator $s$ given the system

For the solution operator, we can compute a condition number given a general $p$-norm (hence, we may associate this number to the problem):

$ kappa (A) = ||A^(-1)||_p dot ||A||_p $

A given problem with a low condition number $kappa$ is said to be well-conditioned (and vice-versa). For most inverse problems, $kappa$ is very high; i.e. its outputs (system parameters) are very sensitive to its inputs (observables). These results motivate the need for regularization methods, part of the next section in this review.

To solve a given inverse problem, an algorithm may be developed: it will be called _backward stable_ if it produces exact solutions to two similar, perturbed inputs (observables), $x$ and $tilde(x)$. Specifically,

$ |x - tilde(x)| <= O(epsilon_m) $

The _backward error_ is related to the _forward error_ associated with our given problem. From our earlier condition number $kappa$ of the problem, we know that

$ |x - x_"true" | <= O(kappa dot epsilon_m) $

Thus, for a well-conditioned problem (low $kappa$), a backward stable algorithm will produce an accurate result.


=== Regularization Methods

With regularization, we may combat Hadamard instability and prevent overfitting. For the latter, parsimony
- prevent
- obtain meaningful results from

obtain meaningful results from observable-sensitive quantities, and also to prevent overfitting, inverse problems must be _regularized_ by modifying the original equations. Common examples include Tikhonov/ridge regularization, truncated SVD (singular value decomposition), and LASSO (least absolute shrinkage and selection operator). Modern methods include [TODO]. In general usage of least-squares, regularization methods are approached as an extra term in the loss function.

$ L(bvec(w)) = ||A bvec(w) - bvec(b)||^2_2 + lambda R(bvec(w)) $

// basically ridge regularization tries to minimize slope (so the resulting model is less sensitive to the input variable -- think about it)

For example, LASSO regularization adopts an $L_1$ norm with $R(bvec(w)) = lambda ||w||_1$, corresponding to a Laplace prior on $bvec(w)$. This type of regularization is often used if we believe the resulting model should have only a small subset of features.

These extra regularization terms introduce a _prior_

Regularization acts in _spectral coordinates_. Tikhonov multiplies each SVD mode by $phi_i (lambda)=sigma_i/(sigma_i^2+lambda)$; truncated SVD applies a hard gate. Thus regularization is best understood as a spectral filter that damps small singular-value modes — the modes most contaminated by noise. This spectral view explains both the mathematical role of $lambda$ and heuristics for choosing $lambda$ (Picard/L-curve/GCV).

// Viewed spectrally, regularization is simply modal filtering: SVD diagonalizes the action of an operator between domain and codomain and exposes the small singular-value modes that amplify noise. Tikhonov and truncated-SVD are different filter prescriptions; L1-based sparsity acts in coefficient space and therefore benefits from SVD/POD preconditioning to reduce correlation and improve identifiability. These connections explain why spectral diagnostics (Picard plots, singular-value decay) are indispensable in designing stable, parsimonious PDE discovery workflows.

=== Sparsity

The Laplace distribution is a sparsity-inducing prior typically used in tasks such as feature selection. However, if applied to

=== Parsimony

The end goal of regularization methods is to penalize complex solutions and balance the trade-off between fitting the data and maintaining stability: this is parsimony (told by stories as Occam's razor). While

On the other end of computational condensed matter physics, there are methods for PDE discovery to accelerate new physical discoveries.

Ginzburg-Landau theory
Discovery of governing equations


== Theory Discovery
Differential equations are the language of physics: they express how systems change over time, space, or other field-specific parameters. For condensed matter physics in particular, hundreds of papers and datasets on spatiotemporal data have been collected.For example, (?).

=== ODE Discovery

=== PDE Discovery

CMP has accumulated hundreds of papers on spatiotemporal data. Take for example, []. First attendants to PDE discovery were []; hinted at PDE discovery.

However, it With the goal of discovering governing theories with the data, researchers were faced with the difficulty of sparse data.

In 2017, Rudy, Brunton, Proctor, and Kutz published a landmark paper opening the field of partial differential equation (PDE) discovery. They had described PDE-FIND, a

Traditionally, a PDE solver

// #align(center)[
//   #diagram(cell-size: 15mm, $
//   	G edge(f, ->) edge("d", pi, ->>) & im(f) \
//   	G slash ker(f) edge("ur", tilde(f), "hook-->")
//   $)
// ]

#align(center)[
  #diagram(edge-stroke: 0.75pt, node-stroke: 0.75pt, {
    node((0, 0), name: <x>)[Input, $arrow(x)$]
    node((0, 1), name: <y>)[Ground truth, $arrow(y)$]
    node((1, 0.5), name: <out>)[MSE]
    let verts = (
      // () means the previous vertex
      ((), "-|", (<y.east>, 50%, <out.west>)),
      ((), "|-", <out>),
      <out>,
    )
    edge(<x>, ..verts, "->") // () == <x>
    edge(<y>, ..verts) // () == <y>
  })
]


Modeling and Simulations


=== Sparse Identification of Nonlinear Dynamics: SINDy


=== Neural ODEs
Adaptation to PDE Discovery Mechanisms

Novel Discovery Mechanisms

== Regime Generalization

Despite the literature,


Introduction
- Motivate why transferability matters
- Could pave the way for a unified theory of superfluidity across different regimes.
