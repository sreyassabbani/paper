#show link: set text(fill: blue)
#show link: underline
#let numeq(content) = math.equation(
  block: true,
  numbering: "(1)",
  content
)

= Oct 27
=== Eigendecomposition (ED) and singular value decomposition (SVD)

- For a matrix $A$, $A^H$ is the _Hermitian conjugate-transpose_. A _Hermitian matrix_ is a matrix with the property that $A^H=A$. If $A$ is real, then a Hermitian matrix is simply a symmetric one $A^T=A$.
- Saw #link("https://math.stackexchange.com/questions/127500/what-is-the-difference-between-singular-value-and-eigenvalue")[this MathSE thread] about _singular values vs eigenvalues_; it had a link to #link("https://www.mathworks.com/content/dam/mathworks/mathworks-dot-com/moler/eigs.pdf")[a chapter] of a book that presented singular values really interestingly (I think; not sure if this was on the videos I skipped for GT math HW 14). 

Given a general (not necessarily square) matrix $A$, non-negative $sigma$, and a pair of nonzero singular vectors $u, v$:

$
cases(
  A u &= sigma v \
  A^H v &= sigma u
)
$

Combining these,
$
A A^H v
&= A (sigma u) \
&= sigma A u \
&= sigma sigma v \
&= sigma^2 v \
A A^H v - sigma^2 v &= 0 \
(A A^H - sigma^2 I) v &= 0
$

So therefore, $A A^H$ must have an eigenvalue of $sigma^2$
