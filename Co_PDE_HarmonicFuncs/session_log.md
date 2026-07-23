# Session Log

## Session 1 — Harmonic Functions Foundations
**Date:** 2026-07-21

### Topics covered
- Meaning of a harmonic function: `Delta u = 0`
- First derivatives as slope/gradient; second derivatives as curvature
- Harmonicity as zero total curvature / zero divergence of the gradient
- Examples and non-examples of harmonic functions
- Importance of the domain and isolated singularities
- Mean value property
- Maximum principle and strong maximum/minimum principle
- Comparison and uniqueness principles
- Harnack inequality: nearby positive harmonic values are comparable
- Liouville-type results for entire harmonic functions
- Removable singularity theorem for bounded harmonic functions on punctured disks

### Demonstrated understanding
- Correctly classified many basic harmonic and non-harmonic functions mentally
- Recognized that harmonicity requires cancellation of directional curvatures
- Understood that a nonconstant harmonic function cannot attain an interior maximum or minimum
- Applied the mean value property to center and circle-average questions
- Applied comparison and uniqueness principles correctly
- Understood that bounded isolated harmonic singularities are removable

### Corrections and misconceptions addressed
- Harmonicity does not mean the gradient is constant
- The gradient may vary while its divergence remains zero
- A positive harmonic function may approach zero at the boundary
- A nonnegative entire harmonic function must be constant
- A bounded harmonic function on a punctured disk cannot oscillate without a limit near the hole
- A condition holding only on a line is insufficient for harmonicity on an open domain

### Learning preference
- Prefer conceptual, mental Q&A without pen-and-paper coefficient calculations
- Avoid long repetitive batches of “find the parameter values” questions
- Use the structure: intuition, assumptions, main tool, proof skeleton, simple example, common confusion, application

### Current checkpoint
The learner has introductory familiarity with harmonicity, maximum/mean-value principles, Harnack intuition, Liouville results, and removable singularities. These topics now need a more structured proof-level treatment.

### Next topic
**Mean value property**
1. Why the center equals the circle average
2. Sphere average versus ball average
3. How the mean value property implies the maximum principle
4. Converse: when the mean value property implies harmonicity
5. Short conceptual verification questions

## Session 2 — Advanced Harmonic Functions and Final Consolidation
**Date:** 2026-07-23

### Topics covered
- Mean value property on spheres and balls, and its role in the maximum principle
- Mean-value inequalities for subharmonic and superharmonic functions
- Strong maximum/minimum principle, comparison principle, and uniqueness for the Dirichlet problem
- Strict interior comparison when two harmonic functions are ordered on the boundary and are not identical
- Radial replacement by averaging over centered circles or spheres
- Classification of radial harmonic functions: `A + B log r` in two dimensions and `A + B r^(2-n)` in dimensions `n >= 3`
- Removable singularities and why boundedness near an isolated harmonic singularity removes the singular radial term
- Unique continuation from a nonempty open subset of a connected domain
- Difference between agreement on an open set and agreement only on a line or hypersurface
- Harmonic analyticity and the distinction between smooth and analytic functions
- Interior derivative estimates of the form `|D^k u(p)| <= C M / R^k`
- Liouville's theorem derived from derivative estimates by sending the available radius to infinity
- Convex composition: `Delta(F(u)) = F''(u)|grad u|^2` when `u` is harmonic
- Consequences for `u^2`, `exp(u)`, and affine compositions
- `L^2` Liouville theorem using subharmonicity of `|u|^2` and growing ball volume
- Harnack chains built from overlapping interior balls
- Local uniform convergence of harmonic series via the Weierstrass M-test
- Positive harmonic series: using Harnack to control every term on a compact set by its value at one point
- Hopf boundary point principle, inward/outward normal derivative signs, and boundary flux interpretation
- Optional Neumann-problem extension: equal normal derivatives determine solutions up to a constant, with zero total flux compatibility

### Demonstrated understanding
- Repeatedly used the standard comparison proof pattern: set `w = u - v`, prove `w` is harmonic, translate the boundary data, and apply the appropriate principle
- Correctly proved uniqueness when two harmonic functions have identical boundary values
- Correctly derived `u <= v` inside from `u <= v` on the boundary
- Recognized that the strong maximum principle upgrades non-strict comparison to strict interior comparison when the functions are not identical
- Correctly explained why equality between the center value and a maximal surrounding average forces all surrounding values to be equal
- Distinguished harmonic, subharmonic, and superharmonic mean-value behavior
- Understood that larger interior distance from the boundary produces stronger derivative control
- Used derivative estimates to explain why bounded entire harmonic functions are constant
- Understood why positivity is essential for multiplicative Harnack comparisons
- Understood how Harnack plus convergence at one point yields local uniform convergence for a series of positive harmonic functions
- Correctly identified unique continuation as propagation of equality throughout the same connected domain
- Correctly identified the inward normal as the gradient direction in the positive-inside, zero-boundary Hopf setting

### Corrections and misconceptions addressed
- Constant radial average does not by itself imply that the original function is bounded or radial
- Boundedness alone does not force an arbitrary function to have a limit; harmonic structure is essential
- Agreement on a line or lower-dimensional set does not imply equality throughout a domain
- A larger admissible radius strengthens, rather than weakens, interior derivative estimates
- Boundedness on a fixed finite ball does not imply constancy because the radius cannot be sent to infinity
- A smooth function need not equal its Taylor series; harmonic functions are stronger because they are analytic
- Boundary equality at some points does not prevent strict inequality at every interior point when the two harmonic functions are not identical
- The ordinary maximum principle gives non-strict comparison; the strong maximum principle supplies strictness
- Hopf's conclusion concerns the normal derivative, not a tangential derivative
- The harmonic-series convergence argument requires one compact-set constant independent of the term index

### Consolidated proof patterns
1. **Comparison and uniqueness:** set `w = u - v`, use linearity of the Laplacian, then apply maximum or strong maximum principles.
2. **Liouville arguments:** obtain an interior estimate on a ball and let the radius grow without bound.
3. **Removable singularities:** isolate the possible radial singular term and use boundedness to eliminate it.
4. **Convex composition:** apply the Laplacian chain rule and use `F'' >= 0`.
5. **Positive harmonic series:** transfer one-point control across compact sets using Harnack, then apply the M-test.
6. **Unique continuation:** use analyticity to propagate equality from an open subset through a connected domain.

### Current checkpoint
All core topics listed in `Co_PDE_HarmonicFuncs/SKILL.md` have now been covered conceptually and consolidated through short proof-pattern questions. The learner can identify the principal theorem needed in standard harmonic-function arguments and can reconstruct the central comparison, Liouville, removable-singularity, Harnack-chain, positive-series, convex-composition, analyticity, derivative-estimate, and Hopf arguments.

### Track status
**Core harmonic-functions track completed.** Future work is optional deepening: formal proofs, worked boundary-value examples, or transition to another PDE topic.
