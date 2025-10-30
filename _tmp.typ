#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node

// double spacing
#set text(top-edge: 0.7em, bottom-edge: -0.3em, size: 11pt)
#set par(leading: 1em)

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

== Introduction

The discovery of governing equations from observational data represents a fundamental challenge at the intersection of physics, mathematics, and computational science. While traditional approaches require deep physical insight and careful theoretical derivation, modern data-driven methods promise to automate aspects of this discovery process. However, a crucial question remains largely unexplored: can PDE discovery methods trained on data from one physical regime generalize to another?

This question holds particular significance for condensed matter physics, where the same physical system—such as a superfluid—may be described by qualitatively different equations depending on temperature, density, and interaction strength. At near-zero temperatures, dilute Bose gases are well-described by the mean-field Gross-Pitaevskii (GP) equation, a nonlinear Schrödinger equation for a single complex wavefunction. At finite temperatures near the critical point, the same system requires Landau's two-fluid hydrodynamics, involving coupled equations for normal and superfluid components with fundamentally different mathematical structure.

The transferability of PDE discovery methods across these regimes could pave the way for several transformative developments. First, it would enable more efficient discovery in data-scarce regimes by leveraging knowledge from well-studied systems. Second, successful transfer would reveal which mathematical structures and physical principles are truly universal versus regime-specific, potentially guiding the development of unified theoretical frameworks. Third, understanding the features that transfer—and those that do not—would illuminate the relationship between emergent and fundamental descriptions in many-body physics.

In the early 1900s, a theory of solid states was taking form due to the discovery of the Drude model of metals (1900), X-ray crystallography (1913), and the discovery of superconductivity (1911). After WWII, the overarching physics grew larger in literature as new theories encompassed more material and phenomena including those of plasmas and liquids, and it quickly adopted a new name: condensed matter physics. Today, condensed matter physics has readily appreciable applications in materials science, quantum computing, and renewable energy technologies, and is often recognized as the most active subfield of physics by researcher count and impact.

Critically, recent computational breakthroughs have been advancing condensed matter physics rapidly. CMP has been influenced by two main types of computational research: forward problems (solving known PDEs) and inverse problems (discovering unknown equations from data), the latter being far newer. This review focuses primarily on inverse problems and the nascent field of cross-regime transferability.

== Background: Physical Regimes in Superfluidity

Understanding the transferability challenge requires a clear picture of the physical regimes under consideration. Superfluidity, the frictionless flow of matter at low temperatures, provides an ideal test case because it exhibits well-characterized regime transitions with corresponding changes in governing equations.

=== The Gross-Pitaevskii Regime

At temperatures near absolute zero and in the dilute limit, Bose-Einstein condensates are accurately described by the Gross-Pitaevskii equation:

$
  i planck.reduce (diff Psi)/(diff t) = (-planck.reduce^2/(2m) nabla^2 + V_"ext" + g |Psi|^2) Psi
$

Here $Psi(bvec(r), t)$ is the macroscopic wavefunction, $m$ is the particle mass, $V_"ext"$ is an external potential, and $g = 4 pi planck.reduce^2 a_s slash m$ characterizes the interaction strength through the s-wave scattering length $a_s$. This is fundamentally a mean-field approximation, valid when interactions are weak ($n a_s^3 << 1$, where $n$ is density) and the temperature is much lower than the critical temperature.

The GP equation is a nonlinear Schrödinger equation with several key mathematical features. It conserves particle number ($integral |Psi|^2 d^3 r$), energy, and exhibits gauge symmetry. It supports topological defects including quantized vortices with circulation $kappa = h slash m$. The healing length $xi = 1 slash sqrt(8 pi n a_s)$ sets the characteristic scale for spatial variations. Dynamically, the system exhibits rich phenomena including solitons, vortex-antivortex pairs, and quantum turbulence—all emerging from a single complex field.

Crucially, the GP equation treats all particles as part of a single quantum condensate. There is no thermal component, no dissipation, and dynamics are fully reversible. This makes it computationally tractable and mathematically elegant, but strictly applicable only at very low temperatures.

=== Two-Fluid Hydrodynamics

As temperature increases toward the critical temperature $T_c$, a significant fraction of particles occupy excited states forming a "normal fluid" component. This regime is described by Landau's two-fluid hydrodynamics, comprising coupled equations for the superfluid and normal fluid densities ($rho_s$, $rho_n$), velocities ($bvec(v)_s$, $bvec(v)_n$), and temperature $T$:

$ (diff rho)/(diff t) + nabla dot (rho_s bvec(v)_s + rho_n bvec(v)_n) = 0 $

$
  rho_s (diff bvec(v)_s)/(diff t) + rho_s (bvec(v)_s dot nabla) bvec(v)_s = -rho_s nabla mu - bvec(F)_"friction"
$

$
  rho_n (diff bvec(v)_n)/(diff t) + rho_n (bvec(v)_n dot nabla) bvec(v)_n = -rho_n nabla mu - nabla P_n + eta nabla^2 bvec(v)_n + bvec(F)_"friction"
$

$
  (diff S)/(diff t) + nabla dot (S bvec(v)_n) = (bvec(F)_"friction" dot (bvec(v)_n - bvec(v)_s))/(T) + "dissipation terms"
$

where $S$ is entropy density, $mu$ is chemical potential, and $bvec(F)_"friction"$ represents mutual friction between the two fluids. The total density is $rho = rho_s + rho_n$.

This framework is fundamentally different from GP in several respects. It involves multiple coupled real-valued fields (compared to one complex field), explicitly includes dissipation through viscosity and mutual friction, and is inherently a phenomenological rather than microscopic description. The superfluid component exhibits irrotational flow ($nabla times bvec(v)_s = 0$ except at vortex cores), while the normal fluid can sustain conventional vorticity. This leads to distinctive phenomena absent in the GP regime: second sound (temperature waves), mutual friction damping of vortices, and thermally-driven counterflow instabilities.

=== The Transferability Challenge

The transition between these regimes presents a formidable challenge for PDE discovery methods. A method trained on GP data encounters a single complex field with conservative dynamics, specific nonlinearity structure, and topological constraints. When applied to two-fluid data, it must somehow adapt to:

*Increased dimensionality:* Moving from 2 real components (Re $Psi$, Im $Psi$) to 6+ fields ($rho_s, rho_n, bvec(v)_s, bvec(v)_n$, etc.), fundamentally changing the search space.

*Different symmetries:* Loss of global U(1) gauge symmetry, different conservation laws (energy no longer conserved due to dissipation), and new constraints (irrotationality of superfluid vs rotationality of normal fluid).

*Emergent vs fundamental structure:* GP derives from microscopic quantum mechanics; two-fluid equations are phenomenological, raising questions about whether data-driven methods can bridge levels of description.

*New physical phenomena:* Dissipative terms (friction, viscosity), thermal effects, and coupled-mode dynamics that have no counterpart in zero-temperature GP.

Conversely, one might ask: what universal features connect these regimes? Both exhibit superfluidity, quantized vortices appear in both (though with different dynamics), and certain scaling relations may persist. Understanding what transfers illuminates which aspects of superfluidity are truly fundamental versus regime-specific.

== Forward Problems

Luo et al. 2025 shows how methods of solving PDEs have evolved over time. "Classical" computational methods (i.e. non-ML or non-data-driven methods) include the finite element/volume/cell method (FEM/FVM/FCM). These discretize space and time, converting PDEs into large systems of algebraic equations. While robust and well-understood, classical methods can be computationally expensive for high-dimensional problems and may struggle with complex geometries or multi-scale phenomena.

Most recently, machine learning is now being applied to solve PDEs efficiently. In particular, physics-informed machine learning is a hybrid approach, incorporating fundamental physical principles on top of data; this addition of principles enhances the model's ability to generalize, especially in scenarios where data is scarce, noisy, or incomplete. Physics-informed neural networks (PINNs), introduced by Raissi et al. 2019, encode PDE constraints directly in the loss function, enabling mesh-free solutions. Neural operators like Fourier Neural Operators (FNO) learn mappings between infinite-dimensional function spaces, offering orders-of-magnitude speedups for parametric PDE families once trained. These methods are known to be very efficient for tackling both forward and inverse PDE problems.

The connection to transferability is significant: neural operators trained on GP equation solutions may partially transfer to two-fluid problems if the underlying mathematical structures (e.g., convection operators, gradient terms) are similar. This motivates investigating whether forward-problem transfer capabilities correlate with inverse-problem (discovery) transfer.

== Inverse Problems

Given a known model $cal(F)_theta$ and data $d$, what parameters $theta$ produced the data?

#align(center)[
  #diagram(
    node-stroke: 1pt,
    edge-stroke: 1pt,
    mark-scale: 60%,
    {
      let (D, T, M) = ((0, 0), (1, 0), (2, 0))
      node(D, [Data $d$])
      node(T, [$theta$?])
      node(M, [$cal(F)_theta$])
      edge(D, T, "->", label: "Infer")
      edge(T, M, "->")
    },
  )
]

For PDE discovery specifically, the inverse problem is more complex: we seek not just parameters but the functional form of the equation itself. This represents a higher level of abstraction, moving from parameter estimation to model selection and symbolic discovery.

=== Hadamard Criteria

The Hadamard criterion for determining whether a particular inverse problem is "well-posed" requires solution existence, uniqueness, and stability; otherwise, it is considered "ill-posed". Inverse problems often fail the third condition, stability, as the behavior of several systems is very sensitive to various parameters; for inverse problems, this indicates that small perturbations in the solution operator (inverse operator) can lead to arbitrarily large variations in the inferred system parameters.

For the solution operator $A$, we can compute a condition number given a general $p$-norm:

$ kappa (A) = ||A^(-1)||_p dot ||A||_p $

A given problem with a low condition number $kappa$ is said to be well-conditioned (and vice-versa). For most inverse problems, $kappa$ is very high; i.e. its outputs (system parameters) are very sensitive to its inputs (observables). These results motivate the need for regularization methods.

To solve a given inverse problem, an algorithm may be developed: it will be called _backward stable_ if it produces exact solutions to two similar, perturbed inputs (observables), $x$ and $tilde(x)$. Specifically,

$ |x - tilde(x)| <= O(epsilon_m) $

The _backward error_ is related to the _forward error_ associated with our given problem. From our earlier condition number $kappa$ of the problem, we know that

$ |x - x_"true" | <= O(kappa dot epsilon_m) $

Thus, for a well-conditioned problem (low $kappa$), a backward stable algorithm will produce an accurate result. For PDE discovery, the conditioning depends critically on the data: measurements of what fields, at what spatial/temporal resolution, with what noise level. Cross-regime transfer exacerbates ill-conditioning because the target regime's operators may differ substantially from the training regime.

=== Regularization Methods

To combat Hadamard instability and prevent overfitting, inverse problems must be _regularized_ by modifying the original equations. Regularization introduces prior knowledge or constraints that restrict the solution space to more physically plausible or mathematically stable solutions. Common examples include Tikhonov/ridge regularization, truncated SVD (singular value decomposition), and LASSO (least absolute shrinkage and selection operator). Modern methods include elastic net (combining L1 and L2), group LASSO (for structured sparsity), and learned regularizers using neural networks.

In general usage of least-squares, regularization methods are approached as an extra term in the loss function:

$ L(bvec(w)) = ||A bvec(w) - bvec(b)||^2_2 + lambda R(bvec(w)) $

where $lambda$ controls the regularization strength. For example, LASSO regularization adopts an $L_1$ norm with $R(bvec(w)) = ||bvec(w)||_1$, corresponding to a Laplace prior on $bvec(w)$. This type of regularization is often used if we believe the resulting model should have only a small subset of active features—a sparsity assumption central to many PDE discovery methods.

Regularization acts in _spectral coordinates_. Tikhonov multiplies each SVD mode by $phi_i (lambda)=sigma_i slash (sigma_i^2+lambda)$; truncated SVD applies a hard cutoff. Thus regularization is best understood as a spectral filter that damps small singular-value modes—the modes most contaminated by noise. This spectral view explains both the mathematical role of $lambda$ and provides heuristics for choosing $lambda$ (Picard plots, L-curve, generalized cross-validation).

The parsimony principle, embodied in Occam's razor, suggests preferring simpler explanations when multiple explanations fit the data equally well. In PDE discovery, this translates to preferring equations with fewer terms or lower-order derivatives. Regularization operationalizes parsimony: the regularization term $R(bvec(w))$ penalizes complexity, and the hyperparameter $lambda$ controls the trade-off between data fidelity and simplicity. For cross-regime transfer, appropriate regularization becomes even more critical—overfitting to source-regime idiosyncrasies prevents generalization to target regimes.

=== Sparsity and Identifiability

The assumption of sparsity—that the true governing equation involves only a small number of terms from a large candidate library—is central to modern PDE discovery. This assumption is physically motivated: fundamental equations of physics tend to be elegant and concise (Maxwell's equations, Navier-Stokes, Schrödinger equation). However, sparsity is basis-dependent: an equation sparse in one representation may be dense in another.

L1 regularization promotes sparsity through its characteristic "corner" in parameter space that encourages exact zeros. However, L1 regularization is not directly equivalent to L0 (counting nonzero elements), and the LASSO solution may include spurious small-magnitude terms or incorrectly zero out important terms when features are correlated. Sequential thresholding, used in methods like SINDy, alternates between regression and hard thresholding to more directly minimize L0 norm.

Identifiability asks: given infinite noiseless data, can we uniquely recover the true equation? For PDE discovery, identifiability is complicated by:

*Gauge freedom:* Multiple equivalent mathematical forms may describe the same physics (e.g., different coordinate systems, different field variables).

*Closure problems:* The data may correspond to a coarse-grained or projected dynamics, not the full microscopic evolution.

*Aliasing:* Insufficient spatial/temporal resolution can make different equations indistinguishable.

*Symmetries:* Conservation laws and symmetries may reduce the identifiable information.

Cross-regime transfer raises new identifiability questions: if a method identifies an equation in the source regime, is that equation the "right" one for transfer, or merely one of several equivalent forms? The choice of basis functions and representation becomes crucial.

== Theory Discovery

Differential equations are the language of physics: they express how systems change over time, space, or other field-specific parameters. Condensed matter physics has accumulated hundreds of papers and datasets on spatiotemporal phenomena—phase transitions, pattern formation, transport phenomena—providing rich ground for automated discovery.

=== PDE-FIND: Sparse Identification of Nonlinear Dynamical Systems

In 2017, Rudy, Brunton, Proctor, and Kutz published a landmark paper introducing PDE-FIND (Partial Differential Equation Functional Identification of Nonlinear Dynamics), opening the modern field of data-driven PDE discovery. The key insight: frame PDE discovery as a sparse regression problem.

The method constructs a library of candidate terms $Theta = [1, u, u^2, u_x, u_(x x), u_t, u u_x, dots]$ from the observed field $u(x,t)$ and its derivatives. If the true PDE has the form:

$ u_t = f(u, u_x, u_(x x), dots) $

then there exists a sparse coefficient vector $bvec(xi)$ such that:

$ bvec(u)_t = Theta bvec(xi) $

where $bvec(u)_t$ is the time derivative computed from data. PDE-FIND solves for $bvec(xi)$ using sequential thresholded least squares: iteratively solve the least-squares problem, threshold small coefficients to exactly zero, and repeat until convergence.

Strengths of PDE-FIND include simplicity, interpretability (recovered equations are explicit), and success on many canonical PDEs (Burgers, KdV, Schrödinger, reaction-diffusion). Limitations include sensitivity to noise (computing derivatives amplifies noise), difficulty with high-dimensional systems, and reliance on choosing a good library (the "library design problem").

For cross-regime transfer, PDE-FIND's library-based approach raises interesting questions: can a library designed for GP equations (with complex-field terms, conservative structure) be adapted for two-fluid equations? Or do we need regime-specific libraries, limiting transferability?

=== SINDy and Extensions

The Sparse Identification of Nonlinear Dynamics (SINDy) algorithm, introduced by Brunton et al. 2016, brought sparse regression to ODE discovery and was subsequently extended to PDEs through weak formulations. Weak-form SINDy avoids computing derivatives directly from noisy data by multiplying the candidate PDE by a test function and integrating, converting derivative estimation into a better-conditioned integral problem.

Recent extensions address various limitations:

*Ensemble SINDy (E-SINDy):* Trains multiple models on bootstrapped data samples, improving robustness to noise and providing uncertainty quantification.

*SINDy with constraints:* Incorporates known conservation laws, symmetries, or positivity constraints directly into the optimization.

*SR3 (Sparse Relaxed Regularized Regression):* Relaxes the hard L0 constraint into a continuous optimization problem, improving convergence.

*TrappingSINDy:* Guarantees global stability properties in the discovered dynamics.

These methods remain fundamentally sparse-regression approaches. Their transferability depends on whether the sparsity pattern (which terms are active) is preserved across regimes—an empirical question for GP versus two-fluid systems.

=== Physics-Informed Neural Networks for Discovery

While PINNs are primarily used as forward solvers, they can be adapted for inverse problems by treating PDE coefficients as learnable parameters. The PINN loss function combines data fidelity and PDE residual:

$ L = L_"data" + lambda L_"PDE" $

$ L_"data" = sum_i |u_"NN"(x_i, t_i) - u_"obs"(x_i, t_i)|^2 $

$ L_"PDE" = sum_j |cal(F)[u_"NN"](x_j, t_j)|^2 $

where $cal(F)$ is the PDE operator with unknown coefficients. By optimizing both network weights and PDE coefficients, PINNs can simultaneously fit data and discover governing equations.

Advantages include handling irregular domains, continuous representation, and implicit regularization through network architecture. Disadvantages include optimization difficulties (loss landscape with many local minima), limited interpretability compared to sparse methods, and tendency to discover complex expressions rather than simple closed forms.

For transfer learning, PINNs offer a natural pathway: pretrain on source regime data, then fine-tune on target regime data with modified physics loss. The network's learned representation may capture transferable features even if the explicit equations differ.

=== Symbolic Regression and Hybrid Approaches

Symbolic regression uses evolutionary algorithms (genetic programming) to search the space of mathematical expressions, directly optimizing for both fit quality and expression simplicity. AI Feynman and related methods apply dimensional analysis and other physics-based heuristics to guide the search, successfully rediscovering fundamental physics equations.

Hybrid methods combine neural and symbolic approaches: use neural networks to learn flexible approximations, then distill them into interpretable symbolic forms. For example, equation learners may use a neural network to learn the right-hand side of an ODE, then apply symbolic regression to find a simple formula matching the network's behavior.

These approaches are promising for cross-regime discovery because they can potentially find entirely new functional forms not anticipated in a fixed library. However, computational cost scales poorly with expression complexity, limiting applicability to simple systems.

== Transfer Learning and Domain Adaptation for PDE Discovery

Transfer learning—leveraging knowledge from a source domain to improve learning in a target domain—has revolutionized machine learning in computer vision and natural language processing. Its application to PDE discovery across physical regimes represents a frontier research area with unique challenges and opportunities.

=== Transfer Learning Paradigms

*Pre-training and fine-tuning:* Train a model on abundant source-regime data, then adapt it to scarce target-regime data by continuing training with adjusted hyperparameters. For PDE discovery, this might mean training a neural network to predict equation coefficients from GP data, then fine-tuning on limited two-fluid data.

*Multi-task learning:* Train a single model to simultaneously discover PDEs across multiple related regimes, encouraging the model to learn shared representations. For superfluidity, this could involve jointly training on GP, two-fluid, and intermediate-temperature data.

*Meta-learning:* Train a model to quickly adapt to new PDE discovery tasks with minimal data. Model-Agnostic Meta-Learning (MAML) and related algorithms optimize for rapid adaptation, potentially enabling few-shot PDE discovery in novel regimes.

*Few-shot learning:* Given only a handful of examples from the target regime, leverage structural knowledge from the source regime. This is particularly relevant for expensive experiments or simulations where target-regime data is scarce.

The key challenge: standard transfer learning assumes similar input-output mappings between domains. For cross-regime PDE discovery, even the dimensionality may change (1 complex field vs 6 real fields), requiring more sophisticated adaptation strategies.

=== Domain Adaptation Techniques

Domain adaptation addresses the problem of distribution shift between source and target domains. Classical techniques include:

*Feature alignment:* Learn representations where source and target data are indistinguishable. Maximum Mean Discrepancy (MMD) and related metrics measure distribution differences in feature space, which can be minimized during training.

*Adversarial domain adaptation:* Use a discriminator network to distinguish source from target features; train the feature extractor to fool the discriminator, creating domain-invariant representations.

*Importance weighting:* Reweight source-domain examples to match target-domain distribution, compensating for covariate shift.

For PDE discovery, domain adaptation might work as follows: extract features from trajectory data (e.g., vortex statistics, energy spectra, correlation functions) that are meaningful in both regimes, then learn to map these features to PDE terms. The features serve as a regime-invariant intermediate representation.

=== Physics-Aware Transfer Strategies

Standard transfer learning techniques often ignore physics structure. Physics-aware approaches exploit domain knowledge:

*Symmetry-based transfer:* Identify symmetries preserved across regimes (e.g., Galilean invariance, rotational symmetry) and enforce them in the learned model. Equivariant neural networks provably preserve specified symmetries.

*Conservation law enforcement:* If certain quantities are conserved in both regimes (mass, momentum), constrain the discovered equations to respect these conservation laws, reducing the effective search space.

*Dimensional analysis and scaling:* Identify dimensionless groups that characterize the physics. The Buckingham Pi theorem ensures that governing equations can be expressed in terms of these groups, potentially simplifying transfer.

*Hierarchical discovery:* First discover universal, regime-independent structures, then add regime-specific corrections. For superfluidity, this might mean learning the vortex core structure (similar in both regimes) before learning dissipation mechanisms (regime-specific).

*Curriculum learning:* Order training regimes from simple to complex. Start with GP (single field, conservative), gradually introduce temperature and normal-fluid coupling, guiding the model through the physics.

The optimal strategy likely combines multiple approaches: use physics constraints to reduce search space, identify transferable features through dimensional analysis, and employ adversarial training to learn regime-invariant representations.

== Methodological Framework for Cross-Regime Transferability

This section outlines a systematic approach to investigate PDE discovery transferability between the Gross-Pitaevskii and two-fluid hydrodynamic regimes.

=== Experimental Design

*Phase 1 - Source training:* Train PDE discovery methods (SINDy, PDE-FIND, PINNs, hybrid approaches) on simulated or experimental GP equation data. Generate diverse training scenarios: different initial conditions (vortices, solitons, turbulence), various trap geometries, different interaction strengths within the GP validity regime.

*Phase 2 - Direct transfer:* Apply source-trained models to two-fluid data without modification. This baseline measures how much transfers "for free" due to shared mathematical structures.

*Phase 3 - Adapted transfer:* Implement adaptation strategies: fine-tune with limited target data, expand the term library, adjust regularization, impose two-fluid-specific constraints (e.g., separate irrotational superfluid from rotational normal fluid).

*Phase 4 - Reverse transfer:* Train on two-fluid data and test on GP data, checking asymmetry in transferability. Two-fluid equations should reduce to GP in the limit $rho_n arrow 0$, $T arrow 0$; does the discovery method recover this limit?

*Phase 5 - Bidirectional learning:* Jointly train on both regimes, investigating whether multi-regime training improves discovery in each individual regime compared to regime-specific training.

=== Evaluation Metrics

Quantitative metrics should assess multiple aspects:

*Prediction accuracy:* Mean squared error between predicted and true dynamics on held-out test data. Separate short-time (captures local dynamics) and long-time (captures stability, attractors) predictions.

*Physical consistency:* Do discovered equations conserve appropriate quantities? For GP: energy, particle number, angular momentum. For two-fluid: mass (but not energy due to dissipation).

*Captured phenomena:* Qualitative checklist of whether discovered equations reproduce:
- Vortex core structure and circulation quantization
- Sound speed (first sound in GP, first and second sound in two-fluid)
- Healing length scale
- Mutual friction effects (two-fluid only)
- Critical velocity for vortex nucleation

*Coefficient accuracy:* For discovered equations of known form, how accurately are coefficients identified? (E.g., if the GP nonlinearity coefficient is known theoretically, does discovery recover it?)

*Sparsity and parsimony:* Number of terms in discovered equations. Simpler is better if accuracy is maintained—prefer equations with physical insight over black-box fits.

*Generalization beyond training conditions:* Test discovered equations on regimes not seen during training (different parameter values, spatial domains, perturbation types).

=== Feature Analysis Framework

To understand what transfers, we must identify features present in both regimes:

*Shared features:*
- Advection structure: $bvec(v) dot nabla$ terms appear in both (velocity advects the field)
- Pressure/chemical potential gradients: Both regimes have $-nabla mu$ or $-nabla P$ terms
- Vorticity dynamics: While details differ, vortex motion is governed by similar topological constraints
- Incompressibility (approximately): In the zero-sound limit, both approach incompressible flow
- Nonlinearity: Both systems are nonlinear, though the specific form differs

*Regime-specific features:*
- GP: Complex field with magnitude and phase degrees of freedom
- GP: Conservative dynamics (Hamiltonian structure)
- GP: Single velocity field
- Two-fluid: Multiple real-valued coupled fields
- Two-fluid: Dissipation (friction, viscosity)
- Two-fluid: Temperature as a dynamical variable
- Two-fluid: Two distinct velocity fields that can have relative motion

By systematically ablating features (e.g., removing dissipative terms from two-fluid data and seeing if GP-trained models perform better), we can identify which features are most important for transfer success or failure.

=== Adaptation Strategies for Cross-Regime Discovery

When direct transfer fails, adaptation can bridge the gap:

*Library expansion:* If GP-trained models use a library $Theta_"GP" = [Psi, |Psi|^2 Psi, nabla^2 Psi, dots]$, expand for two-fluid to include $Theta_"2F" = [rho_s, rho_n, bvec(v)_s, bvec(v)_n, nabla T, "friction terms", dots]$. Retrain with expanded library but initialize coefficients from GP-learned values where terms overlap.

*Hierarchical decomposition:* Separate the equation into transferable (universal) and non-transferable (regime-specific) components:
$ "Full equation" = "Universal part" + "Regime correction" $
Train universal part on source, learn only the correction on target. For example, the vortex advection velocity might be universal, while friction damping is regime-specific.

*Active learning for efficient adaptation:* Strategically choose which target-regime data to collect. Use uncertainty estimates from source-trained models to identify regions of state space where the model is least confident, then prioritize gathering data there.

*Meta-learned initialization:* Instead of random initialization, use meta-learning to find initial parameter values that are "good" starting points for quick adaptation across diverse regimes. Train on a family of related PDEs (GP with varying interaction strength, intermediate-temperature models) to learn to adapt efficiently.

*Physics-informed fine-tuning:* When fine-tuning on target data, add physics-based loss terms specific to the target regime. For two-fluid, add losses penalizing violations of entropy increase, thermodynamic consistency, or known limiting behaviors.

== Expected Challenges and Open Questions

=== Theoretical Challenges

*Emergent vs fundamental equations:* The GP equation derives from microscopic quantum mechanics via mean-field approximation. Two-fluid equations are phenomenological, describing emergent collective behavior. Can data-driven methods bridge levels of description? Or is transfer inherently easier from coarse-grained to fine-grained (or vice versa)?

*Multiple equivalent descriptions:* A given physical system may admit multiple valid mathematical descriptions (different gauge choices, field variables, approximation schemes). Transfer learning might work better for some equivalent forms than others, but without ground truth it's unclear which form is "best."

*Coarse-graining and renormalization:* Moving between regimes often involves coarse-graining—integrating out microscopic degrees of freedom. Standard transfer learning doesn't naturally handle this scale separation. Multiscale methods and renormalization group ideas may inform better transfer strategies.

*Symmetry breaking across regimes:* GP respects U(1) gauge symmetry; two-fluid breaks this (normal and superfluid components are distinguishable). Symmetry-equivariant architectures may fail when the target regime breaks symmetries present in the source.

=== Computational Challenges

*Data scarcity in target regime:* Experiments near critical temperature are difficult; simulations of two-fluid equations are computationally expensive. Transfer learning is motivated by this scarcity, but also hindered by it—not enough target data to properly evaluate adaptation strategies.

*Curse of dimensionality:* GP involves one complex field (2 real components); two-fluid involves 6+ fields. The term library scales exponentially with number of fields, making exhaustive search intractable. Sparsity helps, but may not suffice.

*Noise and measurement limitations:* Real experiments provide noisy, incomplete observations. Some fields (like temperature) may be harder to measure than others (like density). Transfer must be robust to noise distribution shifts between regimes.

*Computational cost of transfer learning:* Meta-learning and multi-task approaches require training across many tasks/regimes, multiplying computational cost. For expensive PDE simulations, this may be prohibitive.

=== Interpretability and Validation Challenges

*Black-box transfer may work but not illuminate physics:* A neural network might successfully transfer predictive capability without providing physical insight. The goal isn't just accurate prediction but understanding what physics is universal versus regime-specific.

*Validation without analytical solutions:* Two-fluid equations generally lack closed-form solutions. How do we validate discovered equations? Comparison to existing phenomenological models is circular (why not just use those models?). Need novel predictions testable by experiment.

*Distinguishing correlation from causation:* Data-driven methods might learn spurious correlations specific to the training distribution. Cross-regime transfer tests generalization, but failure could reflect either fundamental physics differences or mere dataset biases.

*Uncertainty quantification:* How confident should we be in transferred models? Bayesian approaches can provide uncertainty estimates, but computational cost is high. Ensemble methods offer cheaper alternatives but may underestimate uncertainty.

== Future Directions and Broader Implications

The investigation of PDE discovery transferability across physical regimes opens several exciting avenues:

*Automated theory-building:* If successful, transfer learning could accelerate discovery of governing equations in under-explored regimes by bootstrapping from well-studied cases, reducing experimental and computational cost.

*Unified framework for superfluidity:* Understanding what transfers between GP and two-fluid regimes could guide development of a unified theoretical framework that smoothly interpolates between limits and applies across temperatures and interaction strengths.

*Beyond superfluidity:* The methodology generalizes to other phase transitions and regime changes: classical to quantum limits, weak to strong coupling, equilibrium to far-from-equilibrium dynamics, non-relativistic to relativistic regimes.

*Human-AI collaborative discovery:* Rather than fully automated discovery, transfer learning could augment human physicists—proposing candidate equations for expert evaluation, highlighting unexpected regime similarities, or suggesting which experiments would most efficiently discriminate between hypotheses.

*Physics-informed machine learning theory:* This work could contribute to foundational understanding of how physical structure (symmetries, conservation laws, scales) can be exploited for improved transfer learning, informing architecture design beyond physics applications.

The ultimate goal is not to replace theoretical physics with machine learning, but to forge a productive partnership where data-driven methods amplify human insight, accelerating the pace of discovery in an era of increasingly complex and data-rich science.

== Conclusion

The transferability of PDE discovery methods across physical regimes represents a frontier challenge at the intersection of condensed matter physics, machine learning, and computational science. This review has outlined the theoretical foundations, methodological approaches, and open questions surrounding this problem, using the concrete case study of superfluidity across the Gross-Pitaevskii and two-fluid hydrodynamic regimes.

Key insights emerge from this analysis: successful transfer requires careful attention to which physical features are universal versus regime-specific, appropriate mathematical representations that facilitate transfer, and hybrid approaches combining physics knowledge with data-driven flexibility. The challenges are substantial—changes in dimensionality, symmetries, and levels of description pose fundamental obstacles—but the potential rewards are equally significant.

As condensed matter physics generates ever-larger datasets from advanced experiments and simulations, and as machine learning techniques grow more sophisticated, the vision of automated or semi-automated theory discovery becomes increasingly feasible. Understanding what transfers, and what does not, across regimes will be essential to realizing this vision while maintaining the physical insight and interpretability that distinguish science from mere curve-fitting.
