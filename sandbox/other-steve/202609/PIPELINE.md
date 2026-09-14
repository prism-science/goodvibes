# `rigid_body_diffuse_6o2h.ipynb` — pipeline description

One-phonon diffuse scattering from a rigid-body lattice model of triclinic
lysozyme (PDB 6o2h, space group *P*1, one molecule per cell), validated against
synthetic data with a known ground-truth stiffness.

The model: each unit cell holds one rigid body with 6 degrees of freedom, a
rotation $\Omega$ and a translation $v$. Neighbouring bodies are coupled by
harmonic contact springs, each a $6\times6$ stiffness matrix $K$. Solving the
Bloch problem gives the phonons; the diffuse intensity follows from the
one-phonon formula in the classical limit.

---

## 1. Inputs

| File | Contents | Used for |
|---|---|---|
| `6o2h.cif` | atomic coordinates, ADPs | mass matrix, contacts, density phases, ADP comparison |
| `6o2h-sf.cif` | structure factors | measured amplitudes for the hybrid density |

Cell: 27.424 × 32.134 × 34.513 Å, α=88.66° β=108.46° γ=111.88°, V = 26617 Å³.
1151 atoms, 15368 amu, deposited mean *B* = 13.11 Å², ANISOU present for all atoms.

`A_orth` maps fractional to Cartesian coordinates; `B_recip` = $2\pi(A^{-1})^{\mathsf T}$
holds the reciprocal basis vectors as columns, so
$\mathbf q = h\mathbf b_1 + k\mathbf b_2 + l\mathbf b_3$ for real-valued $(h,k,l)$.

---

## 2. Atomic model and mass matrix

Positions, masses, and deposited ADPs are read once. The atomic centre of mass
`r_cm_at` is computed here because **everything downstream references it**: the
mass matrix

$$M=\begin{pmatrix}J&0\\0&mI\end{pmatrix}$$

is block-diagonal only about the centre of mass, and the split of a rigid
displacement $\delta\mathbf r=\mathbf v+\boldsymbol\Omega\times(\mathbf r-\mathbf r_{\rm ref})$
into $(\Omega,v)$ depends on $\mathbf r_{\rm ref}$. If $G$ and $D$ were built about
different points, $I=G^\dagger D^{-1}G$ would contract two incompatible coordinate
systems.

Stiffnesses are carried in units of $k_BT$, so $k_BT=1$ everywhere and no
Boltzmann factor appears in the intensity or displacement formulas. The frequency
unit $\sqrt{k_BT/(\text{amu}\cdot\text{Å}^2)}/2\pi = 2.514$ THz converts reduced
eigenvalues to THz for plotting only.

**The classical limit.** In it, $\langle u_{\mathbf q}u_{\mathbf q}^\dagger\rangle
= k_BT\,D(\mathbf q)^{-1}$, which contains no mass matrix. $M$ therefore affects
*only* the plotted frequencies — never $I(\mathbf q)$, never the ADPs. At 300 K,
$\hbar\omega/k_BT = 0.16\times(\omega/\mathrm{THz})$, and the branches here sit
below ~0.5 THz, so the classical form is amply justified.

---

## 3. Electron density

### 3.1 Why the density and not atomic form factors

$F$ and $L$ come from the **hybrid map** — experimentally measured amplitudes with
phases from the refined model — rather than from an isolated-atom form-factor
table. Two reasons:

1. It is a more faithful estimate of the crystal's average density: bonding
   density, ordered solvent, and real static and dynamic disorder are all present.
2. More importantly, it is the *correct* density. The hybrid map is the
   crystallographic **average** density, i.e. the instantaneous molecular density
   convolved with the displacement distribution: $F_{\rm avg}=F_0e^{-W}$. The
   one-phonon intensity requires $|F_0|^2e^{-2W}$, which is exactly
   $|F_{\rm avg}|^2$. The Debye–Waller factor therefore enters automatically, with
   no separate bookkeeping and no decision about whether to apply the deposited
   ADPs to a calculated transform.

The grid is sized from `D_MIN_TRANSFORM = 2.0 Å` at `RATE_TRANSFORM = 2.5`, giving
36×48×48 = 82944 voxels at ~0.7 Å spacing. The transform cutoff is deliberately
finer than the resolution at which the diffuse model is used, because masking one
molecule out of a periodic map convolves its transform with the mask transform and
so mixes nearby $|q|$.

The grid dimensions are taken from the **actual reflection indices**
($n_i \ge 2h_{\max,i}+1$, here $|h|_{\max} = (13,16,17)$) rather than from
$a_i/d_{\min}$, which under-counts $h_{\max}$ for a triclinic cell and would alias
distinct reflections onto the same grid point.

$F(000)$ is not measured, so the map integrates to zero: it is the true density
minus the **cell average** $\bar\rho = Z_{\rm cell}/V$, which is *not* the same as
the bulk solvent level.

### 3.2 Unwrapping onto a single molecule

The grid holds the **crystal** density folded into one box. Transforming it
directly gives

$$F_{\rm cell}(\mathbf q)=\sum_{\mathbf n}e^{-i\mathbf q\cdot\mathbf T_{\mathbf n}}
\int_{\rm box-\mathbf T_{\mathbf n}}\rho_{\rm mol}(\mathbf r')e^{-i\mathbf q\cdot\mathbf r'}d^3r'$$

At integer $hkl$ the phase factors are all 1 and $F_{\rm cell}=F_{\rm mol}$. At
the **fractional** $hkl$ this pipeline runs on they are not, and the parts of the
molecule folded across a cell boundary return with the wrong phase. This is not
fixable by rolling the grid: a protein's molecular region tiles space, so no
translation of the box leaves empty margins.

Each voxel is therefore assigned to the molecular image it belongs to — nearest
**atom surface** (centre distance minus van der Waals radius, so a large atom does
not lose surface voxels to a small neighbour) over all lattice images within
`UNWRAP_IMAGE`. That assignment is a fundamental domain: one voxel per lattice
orbit, nothing lost or double-counted. Each voxel gains an integer translation
$\mathbf n_v$, and its position in the molecule's own frame is
$\mathbf r_v-\mathbf T_{\mathbf n_v}$, giving

$$F_{\rm mol}(\mathbf q)=\sum_v\rho_v\,e^{-i\mathbf q\cdot\mathbf r_v}\,
e^{+2\pi i\,(h,k,l)\cdot\mathbf n_v}\,dV$$

The correction is a per-voxel phase, exactly 1 at integer $hkl$. Since
$\mathbf n_v$ takes only ~15 distinct values, the sum splits into one ordinary
transform per translation group.

For this crystal 60% of voxels stay at $\mathbf n=(0,0,0)$, 21% go to $(0,0,-1)$,
and the rest are scattered over 13 further images. The unwrapped molecule spans
35×44×54 Å against a 27×32×35 Å cell — larger, as it must be.

**Sufficiency check.** A large share of voxels in *non-zero* images is expected and
harmless; the meaningful test is whether widening the search by one shell would
claim any voxels at all. The assignment is therefore repeated with
`UNWRAP_IMAGE+1` and the fraction landing beyond the original radius is reported.

### 3.3 Bulk solvent: the excluded-volume contrast

Write the crystal density as protein plus flat bulk solvent filling the complement
of the molecular envelope $M$:

$$\rho_c=\underbrace{\rho_{\rm sol}}_{\text{static, uniform}}
+\sum_{\mathbf n}\underbrace{\bigl[\rho_{\rm prot}-\rho_{\rm sol}M\bigr]}
_{\textstyle\rho_{\rm eff},\ \text{moves rigidly}}(\mathbf r-\mathbf T_{\mathbf n})$$

When a molecule displaces, its atoms move **and so does the hole it occupies in
the solvent** — water rearranges behind it far faster than a phonon period. The
uniform bulk does not move. So the object generating one-phonon scattering is the
**contrast** density $\rho_{\rm eff}$, the same quantity that governs solution
scattering.

**Negative regions are correct.** Inside the envelope, wherever the protein's own
density falls below the bulk solvent level — interatomic voids, and the shell
between the flat-solvent boundary and the first atoms — the contrast is negative:
those regions hold fewer electrons than the solvent they displaced. 58% of voxels
inside the envelope are negative here. The first diagnostic panel plots the
profile along the distance-from-atom-surface axis, and it should show a flat
plateau at $\rho_{\rm sol}$, a dip **below** it, then a rise to the atomic peaks.

**Babinet.** The solvent occupies the complement of the envelope, and
$\mathcal F[1]=\delta(\mathbf q)$, so at $\mathbf q\neq0$

$$\mathcal F\bigl[\rho_{\rm sol}(1-M)\bigr]=-\rho_{\rm sol}\,\mathcal F[M]$$

Complementary regions scatter identically apart from the direct beam. That is what
allows solvent spread over the whole complement to be represented as a *negative*
density confined inside the envelope, making the moving object compactly
supported. It is the same flat bulk-solvent model used in refinement,
$F_{\rm sol}=-k_{\rm sol}e^{-B_{\rm sol}s^2}F_{\rm mask}$. Transforming a density
with negative regions needs nothing special — the Fourier transform is linear.
Babinet justifies the *form* of $\rho_{\rm eff}$, not its evaluation.

**Implementation.** $\rho_{\rm sol}$ is read off the map itself as the median over
deep solvent (beyond `SOLVENT_PROBE + SOLVENT_SHELL` from any atom surface), so no
assumption about absolute scale enters. The envelope is the standard bulk-solvent
mask — within `SOLVENT_PROBE = 1.1 Å` of the van der Waals surface — softened with
a `tanh` ramp of width `SOLVENT_EDGE = 0.7 Å`, the real-space counterpart of the
$B_{\rm sol}$ smearing applied to $F_{\rm mask}$ in refinement; a hard edge would
ring in Fourier space. Then $\rho_{\rm eff}=(\rho-\rho_{\rm sol})\cdot M_{\rm soft}$.

The envelope also makes the image-assignment boundary harmless: it lies in bulk
solvent where $M\approx0$, so ambiguously-assigned voxels contribute nothing. At
crystal contacts the envelopes of neighbouring molecules overlap and the
nearest-surface rule splits the shared density between them.

**Absolute-scale check.** The map is $\rho_{\rm true}-\bar\rho$ with
$\bar\rho=Z_{\rm cell}/V$, and the cell holds protein plus solvent:
$Z_{\rm cell}=Z_{\rm prot}+\rho_{\rm sol}^{\rm abs}V_{\rm solv}$. With
$\rho_{\rm sol}^{\rm abs}=\rho_{\rm sol}^{\rm map}+\bar\rho$ that pair solves to

$$\bar\rho=\frac{Z_{\rm prot}+\rho_{\rm sol}^{\rm map}V_{\rm solv}}{V_{\rm mask}}$$

For this run: $\rho_{\rm sol}^{\rm map}=-0.0527$, $V_{\rm mask}=20793$ Å³,
giving $\bar\rho = 0.355$ and $\rho_{\rm sol}^{\rm abs}=0.302$ e⁻/Å³ against
0.335 for bulk water — low by about the 12% of electrons the hydrogen-free
deposited model omits.

**Consistency.** Since the map integrates to zero over the cell,
$\int\rho_{\rm eff}dV=-\rho_{\rm sol}^{\rm map}V_{\rm cell}$ holds exactly *if* the
map is flat at $\rho_{\rm sol}$ throughout the solvent region. Measured 1532 e⁻
against 1403 e⁻ predicted: an 8% gap, which measures how well that flat-solvent
assumption holds.

**Scale consequence.** Contrast electrons are 1532 e⁻ against 7690 e⁻ raw — a
factor ~5 in $|F|$, ~25 in intensity at low $q$. The effect is confined to low
resolution: at high $hkl$ the envelope transform is negligible and
$F_{\rm eff}\to F_{\rm meas}$, which the resolution-binned check confirms.

---

## 4. Molecular transform and the coupling vector

Two evaluation paths, both exact:

* **`F_L_batch`** — arbitrary scattered $(h,k,l)$, one direct sum per translation
  group, chunked against a memory budget.
* **`F_L_plane`** — a full $(h,k)$ plane by three sequential axis contractions.
  $q\cdot r$ factorizes on a lattice grid, so contracting one axis at a time costs
  $O(N_{\rm vox}+n_Hn_un_v+n_Hn_Kn_v)$ instead of $O(n_Hn_KN_{\rm vox})$ — roughly
  three orders of magnitude, which is what makes full-map evaluation affordable.

Weight channels $[\rho_{\rm eff},\rho_{\rm eff}(x-x_{\rm cm}),\dots]\times dV$ are
packed into one matrix so a single product per group yields $F$ and all three
components of $L$.

$$G=\begin{pmatrix}G_R\\G_T\end{pmatrix}
=\begin{pmatrix}i\mathbf q\times L(\mathbf q)\\i\mathbf q\,F(\mathbf q)\end{pmatrix}$$

$G$ is complex and stays complex; $I=G^\dagger D^{-1}G$ is a Hermitian quadratic
form in a complex Hermitian $D$.

**Three verifications.**

1. At integer $hkl$, $|F|/|F_{\rm meas}|$ binned by resolution — approaches 1 at
   high resolution, falls below at low resolution where the solvent subtraction
   removes amplitude the bulk contributed.
2. At fractional $hkl$, the grouped construction against brute-force summation over
   explicitly unwrapped coordinates: agreement to $10^{-15}$ relative. A transform
   of the periodic grid without the unwrapping phase is evaluated alongside and
   differs by 131%, which is the size of the effect the phase corrects.
3. Plane path against scattered path: $10^{-16}$ relative.

---

## 5. Contacts, prior, and parameterization

### 5.1 Contact detection

Two cell images are in contact if any atoms fall within `CONTACT_CUTOFF = 4.0 Å`.
Born–von Kármán translational symmetry means the contact depends only on
$\mathbf n$, and $\mathbf n$ and $-\mathbf n$ are the same physical spring seen
from either side — so 12 directed images collapse to 6 independent stiffnesses.

This is **not** a point-group symmetry reduction: *P*1 has no point symmetry, so
the six are genuinely *distinct* and each carries its own free $6\times6$ matrix.
`MAX_IMAGE = 1` is verified by re-running the search one shell wider.

Detected here:

| $\mathbf n$ | $\lvert R_n\rvert$ (Å) | atom pairs | patch $\rho_g$ (Å) |
|---|---|---|---|
| (−1,−1,0) | 33.58 | 33 | 6.43 |
| (−1,0,−1) | 36.66 | 16 | 6.71 |
| (−1,0,0) | 27.42 | 31 | 10.37 |
| (0,−1,0) | 32.13 | 63 | 6.78 |
| (0,−1,1) | 46.60 | 9 | 3.47 |
| (0,0,−1) | 34.51 | 53 | 5.89 |

### 5.2 The geometric prior

Model each contact as $n$ independent isotropic point springs of stiffness $k$ at
the atom-pair midpoints $\mathbf m_i$, with $\mathbf d_i=\mathbf m_i-\bar{\mathbf m}$:

$$E=\tfrac{k}{2}\sum_i\bigl|\mathbf v+\boldsymbol\Omega\times\mathbf d_i\bigr|^2
\;\Longrightarrow\;
K_{TT}=knI_3,\quad
K_{RR}=k\sum_i\bigl(|\mathbf d_i|^2I-\mathbf d_i\mathbf d_i^{\mathsf T}\bigr),\quad
K_{TR}=0$$

the last exactly, since $\sum_i\mathbf d_i=0$ at the centroid. With no extra
parameters this supplies:

* **contact-count scaling** — $\kappa_T$ runs 9.0 to 64.0 $k_BT$/Å² across the six,
  tracking the 9-to-63 spread in atom pairs;
* **the $\kappa_R/\kappa_T$ ratio**, of order $\rho_g^2$ — 8 to 69 Å² here. The two
  are not interchangeable: one is $k_BT/\mathrm{rad}^2$, the other $k_BT/\mathrm{Å}^2$,
  and setting both to the same number makes the librational springs too soft by
  $\rho_g^2$;
* **patch anisotropy** — a flat contact is automatically soft about axes in its own
  plane;
* **a vanishing local-frame $T$–$R$ block**, rather than one imposed by hand.

`PRIOR_K_PER_PAIR = 1.0` $k_BT$/Å² = 0.41 N/m per atom pair, the right order for a
weak non-covalent contact (vdW ~1 N/m, H-bond ~10–100 N/m). Only the prior's
*shape* is load-bearing; the magnitude is what the refinement adjusts.

### 5.3 Parameterization

$K$ is specified in a **local contact frame** (origin at the measured contact
centroid, $z$ along $\mathbf R_{\mathbf n}$) and re-expressed at the far body's own
centre of mass by a fixed transform computed once from the structure.

$K=LL^{\mathsf T}$ with $L$ lower-triangular and otherwise free is positive
semidefinite for any real $L$, so the optimizer runs unconstrained: 6 contacts ×
21 Cholesky entries = **126 free parameters**.

---

## 6. Dynamical matrix

$$D_{\mathbf n}(\mathbf q)=M_{\mathbf n}(\mathbf q)^\dagger K_{\mathbf n}M_{\mathbf n}(\mathbf q),
\qquad M_{\mathbf n}(\mathbf q)=A_{\mathbf n}-e^{i\mathbf q\cdot\mathbf R_{\mathbf n}}I$$

with $A_{\mathbf n}$ the rigid-shift adjoint, summed over the 6 contacts. This
factored form keeps $D$ exactly Hermitian positive semidefinite in floating point,
so cancellation cannot push an acoustic eigenvalue negative — and it makes
$K\succeq0\Rightarrow D\succeq0$ immediate rather than approximate.

$D$ is complex Hermitian, not real symmetric:
$\mathrm{Im}\,D_{\mathbf n}=\sin(\mathbf q\cdot\mathbf R_{\mathbf n})(KA-(KA)^{\mathsf T})$,
which runs ~1.6% of $\|\mathrm{Re}\,D\|$ here.

**Zero modes at Γ.** $D(\mathbf q)u=0$ iff $M_{\mathbf n}(\mathbf q)u=0$ for every
$\mathbf n$. At $\mathbf q=0$ that reads $\mathbf R_{\mathbf n}\times\Omega=0$ for
all $\mathbf n$, which for three independent $\mathbf R_{\mathbf n}$ forces
$\Omega=0$ with $v$ free: **exactly 3** zero modes, the pure translations. The
three librational eigenvalues sit at whatever rest frequency the rotational
stiffness sets (0.065, 0.072, 0.093 in reduced units for the prior), and are *not*
required to vanish. A BvK lattice with fixed $\mathbf R_{\mathbf n}$ is not
rotationally invariant, so a global crystal rotation is not a zero mode of this
model — a limitation worth naming.

**Intensity.** $I(\mathbf q)=G^\dagger D^{-1}G$. $D$ is periodic in the reciprocal
lattice, so $D(\mathbf G)=D(0)$ at every Bragg position: the acoustic eigenvalues
vanish there and a pseudo-inverse keeps $I$ finite. Off a Bragg spot the acoustic
eigenvalues grow as $|\delta\mathbf q|^2$, producing the
$I\sim|\mathbf q-\mathbf G|^{-2}$ halos.

Halo brightness is modulated by $|G|^2\approx|\mathbf qF(\mathbf q)|^2$ — weak
where the molecular transform is weak, growing as $|\mathbf q|^2$ overall. Not by
systematic absences: *P*1 has none, and the acoustic branches vanish at every
reciprocal-lattice point.

**Cross-path assertion.** Two independent code paths evaluate $I$ — the
refinement's linear solve and the map/ADP eigendecomposition with pseudo-inverse.
They are asserted to agree ($4\times10^{-13}$ here). This is a check the synthetic
ground truth cannot supply, since an error common to both would cancel.

---

## 7. Synthetic data

The ground truth is the geometric prior perturbed by a random multiplicative
distortion of its Cholesky factor (`distortion = 0.35`). Since the prior is also
the refinement's starting point, this asks whether the data can move the fit a
realistic distance in a realistic direction, rather than whether it can recover a
target chosen for convenience. The perturbation is substantial: eigenvalue spreads
widen by roughly an order of magnitude relative to the prior.

Sampling is stratified — a near-Bragg subset (acoustic-dominated, sensitive to the
long-wavelength elastic limit) plus a larger mid-zone subset (sensitive to the
individual contact stiffnesses), over two $l$ layers, with Bragg positions
excluded. A conditioning screen at the true $K$ discards points where $D$ is
near-degenerate and the plain linear solve would be ill-conditioned (20 of 300
here).

Noise is Gaussian with $\sigma = \text{floor} + 0.06\sqrt{I}$; 15% of points are
held out. Weights are $1/\sigma^2$, and the **effective sample size**
$(\sum w)^2/\sum w^2$ is reported alongside the nominal count (221 of 246 here),
because a nominal count means nothing once weights are heterogeneous.

---

## 8. Refinement

$$\mathcal L(\theta)=\sum_{\mathbf q\in\rm train}w(\mathbf q)
\bigl[I_{\rm obs}-I_{\rm model}(\theta)\bigr]^2
+\lambda\sum_{\mathbf n}\bigl\|\log\bigl(K_{\rm prior}^{-1/2}K_{\mathbf n}
K_{\rm prior}^{-1/2}\bigr)\bigr\|_F^2$$

### 8.1 The regularizer metric

$K$ mixes units: $K_{RR}$ is $k_BT/\mathrm{rad}^2$, $K_{TT}$ is $k_BT/\mathrm{Å}^2$,
$K_{RT}$ is $k_BT/(\mathrm{rad\,Å})$. A Frobenius norm of $K-K^{\rm prior}$ would
add squared quantities differing by powers of length, and would weight the
rotational block more heavily by $\sim\rho_g^4$ — measured on contact (−1,−1,0),
**938×**, which leaves the translational stiffnesses effectively unregularized.

The penalty is the **affine-invariant (geodesic) distance** on the cone of
positive-definite matrices,

$$d^2=\bigl\|\log\bigl(K_{\rm prior}^{-1/2}KK_{\rm prior}^{-1/2}\bigr)\bigr\|_F^2
=\sum_i(\log s_i)^2$$

with $s_i$ the generalized eigenvalues of the pencil $(K,K^{\rm prior})$. The
prior carries exactly the units of $K$, so the ratio is dimensionless and no
length scale has to be invented. Three further properties:

* **Coordinate-invariant** under $K\to TKT^{\mathsf T}$, which is what moving the
  reference point from the contact centroid to the centre of mass, or rotating the
  local axes, amounts to. The penalty therefore does not depend on a bookkeeping
  choice; a Frobenius norm does, by ~0.7× on a test contact.
* **Symmetric in stiff/soft** — a factor of two too stiff costs exactly what a
  factor of two too soft costs. A Frobenius norm caps the softening penalty at
  $\|K^{\rm prior}\|_F^2$ as $K\to0$ while leaving stiffening unbounded.
* **Barrier at the boundary** — $d\to\infty$ as $K$ approaches singular, keeping
  each contact positive definite without a bound constraint.

The notebook prints the block balance: equal 10% relative perturbations of the
rotational and translational blocks cost the same to 3 decimal places.

### 8.2 Gradient

$I=G^\dagger D^{-1}G$ and $d(D^{-1})=-D^{-1}dD\,D^{-1}$, so $dI=-v^\dagger dD\,v$
with $v=D^{-1}G$. $D$ is linear in each $K_{\mathbf n}$, so $dI/dK_{\mathbf n}$ is
closed-form at essentially no cost beyond the forward evaluation. The regularizer
contributes $2\lambda K_{\rm prior}^{-1/2}(P^{-1}\log P)K_{\rm prior}^{-1/2}$ with
$P=K_{\rm prior}^{-1/2}KK_{\rm prior}^{-1/2}$. Both chain through the fixed
lab-frame transform and $K=LL^{\mathsf T}$ to give $d\mathcal L/dL$, checked
against central finite differences (max relative error ~$10^{-7}$).

$G$ depends only on the structure and $\mathbf q$, never on $\theta$, so it is
cached once rather than recomputed at every iteration.

### 8.3 Convergence: two questions, two tests

**Is the fit converged?** Once the loss stops improving by more than about one
$\chi^2$ unit there is nothing left to learn: the noise on $\chi^2$ itself is
$\sqrt{2\,\mathrm{dof}}\approx15$ here, so smaller changes cannot alter any
statement about the model. A trailing-window test on the loss — improvement below
`LOSS_TOL` over `LOSS_WIN` consecutive segments — is the stopping rule.

**Is $K$ converged?** A different question. Along a flat direction the loss barely
changes while $K$ travels a long way, so both the projected gradient and the loss
can look settled while the stiffnesses are still moving. The **geodesic step**
measures that directly, in the same metric as the regularizer: 0.01 is a 1% change
in one stiffness eigenvalue of one contact.

L-BFGS-B runs in warm-started segments of `BLOCK` iterations with `ftol` and
`gtol` set to zero inside, so each runs its full block and both quantities can be
measured between them. The loss is reported split into data and penalty, so it is
comparable across $\lambda$. Segmenting costs a few percent of iterations to
curvature rebuilds.

Two failure modes, both reported explicitly:

* A gradient-based stop is a false positive when the valley is flat. A run at
  $\lambda=10^{-2}$ reported scipy status 0 at iteration 4515 while $K$ was moving
  5.8e-3 per iteration and *accelerating* — 2.5e-3 over the second half, 4.0e-3
  over the last 10%, 5.8e-3 over the last 1%.
* When the loss criterion fires while the geodesic step is still large, the fit is
  statistically converged but $K$ is not: the reported stiffnesses are one point
  in a nearly flat region rather than a point estimate. That region is what the
  prior and the sloppiness spectrum describe.

A representative run at $\lambda=0.3$: the geodesic step decayed as $n^{-0.87}$,
a factor of 14 over 19000 iterations, confirming the regularizer now supplies a
restoring force. But it would need $\sim2.6\times10^6$ iterations to reach
$10^{-3}$, while the loss had already flattened — the last 4500 iterations moved
it by 0.18, some 85 times below the noise on $\chi^2$. Hence the loss-based stop,
which on that trace fires around 7500 iterations leaving 1.4 $\chi^2$ units
unclaimed.

### 8.4 Choosing `LAMBDA_PRIOR`

The regularizer weight decides whether the poorly-determined directions have a
restoring force at all. At $\lambda=10^{-2}$ the fit travelled 6.41 geodesic units
from the prior — a factor of $e^{6.41}\approx600$ along the worst direction —
while the ground truth sits only about 2 units away, and the penalty contributed
under 1% of the loss.

$\lambda$ is selected by a scan on **held-out $R$**, which needs no ground truth
and so transfers unchanged to the experimental notebook. The scan reports, per
$\lambda$: $\chi^2/\mathrm{dof}$, held-out $R$ and $CC$, the geodesic distance
prior→fit, the iteration count, and the final geodesic step — the last showing
directly whether the problem has become well-posed. In the synthetic notebook the
distance truth→fit is reported alongside as independent confirmation, and
prior→truth is marked on the plot.

The earlier suggestion to place $\lambda$ at the noise floor of the Gauss-Newton
spectrum does not transfer between metrics: those eigenvalues live in
Cholesky-parameter space while $\lambda$ multiplies a geodesic distance in
$K$-space. The held-out scan is the reliable route.

---

## 9. Sloppiness

126 parameters, and no reason to believe the data constrains 126 independent
combinations. The Gauss-Newton Hessian $J^{\mathsf T}J$, from the weighted
Jacobian $J_{ij}=\sqrt{w_i}\,\partial I_i/\partial\theta_j$ at the optimum, answers
this directly.

For this run **72 of 126** directions sit above a $10^{-6}$ noise floor; the other
54 are set by the prior. That number is what should be quoted alongside a fitted
$K$, rather than 126 apparently-measured values. It also gives a principled value
for `LAMBDA_PRIOR`: place it near the spectrum's noise floor, where it controls
everything the data does not and nothing it does.

The spectrum also explains the asymmetry visible in the band-structure convergence
movie — acoustic branches lock on quickly, librational ones drift — since those
are the stiff and sloppy directions respectively.

---

## 10. Atomic displacement parameters

$$\Sigma=\langle(\Omega,v)(\Omega,v)^{\mathsf T}\rangle
=\frac{k_BT}{N}\sum_{\mathbf q}D(\mathbf q)^{-1}$$

projected onto each atom by $J_r$ to give the full $3\times3$ tensor
$U=J_r\Sigma J_r^{\mathsf T}$ — the same object an ANISOU record holds — with
$B=\tfrac{8\pi^2}{3}\mathrm{Tr}(U)$.

Two quadrature details matter. The grid is **offset** (Monkhorst–Pack style) so it
never lands on Γ: the acoustic contribution goes as $\int d^3q/q^2$, convergent in
three dimensions but easy to undersample with a coarse Γ-inclusive grid. And the
eigenvalue floor is **absolute** rather than relative to $\mathrm{ev}_{\max}$ at
each $q$, since with a strongly anisotropic $K$ a genuine small acoustic eigenvalue
near Γ could otherwise be zeroed merely because $\mathrm{ev}_{\max}$ is large
there. Convergence in `n_grid` is checked explicitly rather than assumed.

Both models are in the same reduced units, so the atom-by-atom comparison of $B$
or of the full $U$ is meaningful on an absolute scale. The deposited ADPs are
shown separately: against an arbitrarily-chosen synthetic $K$ an absolute
comparison carries no meaning, and only becomes direct once $K$ is fit to real
diffuse data.

The anisotropic comparison is the more informative of the two, because unlike the
isotropic trace it cannot be reproduced by the rigid-body lever arm alone.

---

## 11. Visualization

Phonon modes at a chosen $\mathbf q$ are rendered as looping GIFs of the rigid-body
motion. A phonon eigenvector at general $\mathbf q$ is genuinely complex — the
physical motion is $\mathrm{Re}(e\,e^{i(\mathbf q\cdot\mathbf R-\omega t)})$, so
components lead each other in phase. For a single-cell animation each mode is
rotated by the global phase that makes it as real as possible (exact for a
standing mode, an approximation otherwise) and the residual out-of-phase amplitude
is reported per mode.

Mode character is labelled by the rotational kinetic-energy fraction of the
mass-weighted eigenvector. Displacement amplitude is the largest per-atom
displacement in Å, not the centre-of-mass displacement.

---

## 12. Key parameters

| Parameter | Value | Role |
|---|---|---|
| `D_MIN_TRANSFORM` | 2.0 Å | transform density resolution; finer than the diffuse cutoff by design |
| `RATE_TRANSFORM` | 2.5 | grid oversampling |
| `UNWRAP_IMAGE` | 1 | lattice images searched for voxel assignment; verified |
| `SOLVENT_PROBE` | 1.1 Å | excluded-volume envelope beyond the vdW surface |
| `SOLVENT_EDGE` | 0.7 Å | tanh softening width of the envelope |
| `SOLVENT_SHELL` | 1.5 Å | extra margin for reading the flat solvent level |
| `CONTACT_CUTOFF` | 4.0 Å | atom–atom cutoff defining a contact |
| `MIN_CONTACTS` | 5 | minimum atom pairs for a contact to count |
| `MAX_IMAGE` | 1 | contact search radius; verified |
| `PRIOR_K_PER_PAIR` | 1.0 $k_BT$/Å² | prior spring stiffness per atom pair |
| `LAMBDA_PRIOR` | 1e-2 | regularizer weight; set from the Gauss-Newton spectrum |
| `REG_EIG_FLOOR` | 1e-10 | eigenvalue floor in the matrix logarithm |
| `BZ_NGRID` | 20 | Brillouin-zone grid for $\Sigma$; convergence checked |
| `D_MIN_DIFFUSE` | 3.0 Å | resolution cutoff for the diffuse maps |

---

## 13. What this notebook does and does not establish

**Establishes.** The refinement machinery works: the analytic gradient is correct,
the optimizer recovers a substantially perturbed ground truth from noisy data
($R=0.008$ training, $R=0.017$ held out), the recovered ADPs match, and the
sloppiness analysis quantifies how much of the 126-dimensional parameter space the
data actually constrains.

**Does not establish.** That the forward model is right. Generating and fitting
with the same code cancels any error common to both — the ADP comparison in
particular is insensitive to a systematic error in $\Sigma$. The forward model is
checked separately, by assertions independent of the ground truth: the transform
against brute-force summation and against the measured amplitudes, the two
intensity paths against each other, and the Γ-point mode count against the
analytic prediction.

**Does not address.** Whether a single-rigid-body model with harmonic contacts is
adequate for real lysozyme. Internal motion, side-chain disorder, and solvent
dynamics are all outside it, and only a fit to experimental data can say how much
of the observed diffuse signal they carry.


---

## Appendix: the experimental notebook

`rigid_body_diffuse_6o2h_experimental.ipynb` fits the same model to the deposited
diffuse maps (CXIDB 128, `triclinic_lysozyme_maps.h5`). Sections 2–6 above —
atomic model, density, unwrapping, solvent contrast, transform, contacts, prior,
parameterization, dynamical matrix — are **byte-identical** between the two
notebooks. What differs:

**Data.** Instead of generating intensities, one streaming pass over the HDF5 map
collects candidate voxels and resolution diagnostics simultaneously, holding one
$(n_k\times n_l)$ plane in memory at a time. `D_FIT_MIN`/`D_FIT_MAX` are set by
model validity rather than by an SNR heuristic, since mean$(I)$/mean$(\sigma)$ per
shell climbs with resolution because the numerator grows and would always select
the finest shell — the wrong end, where internal motion dominates and the
one-phonon approximation degrades.

**Sampling.** A halo profile (shell × angular-sector averages around every
reciprocal-lattice point, beating the noise down by $\sim\sqrt N$ and reading out
the acoustic limit directly) plus an equal-count stratified mid-zone sample over a
resolution × Bragg-distance grid. The halo preference is expressed through
*sampling* rather than a multiplicative weight, because the voxel population grows
as $d^2$ and a short-length-scale weight would collapse the effective sample size.
Negative intensities are kept — filtering on the sign of the fitted quantity would
bias the result — and the effective sample size is printed with every count.

**Scale.** $D$ is linear in $K$, so $\{K,s\}\to\{\lambda K,\lambda s\}$ is exactly
degenerate. The degenerate direction is removed from the parameter space by
refining the normalized *shape* of $K$ with $\sum_{\mathbf n}\mathrm{tr}K_{\mathbf n}$
fixed and $s$ profiled in closed form, then the gauge is closed with units:
$I_{\rm model}$ with $k_BT=1$ and $G$ in $e/$Å is in electrons² per unit cell, the
same scale the deposited map reports, so $s\equiv1$. **No ADP information enters
the fit.**

**Validation.** The deposited ADPs are the independent check. The predicted mean
$B$ should fall *below* the deposited mean, since the latter also contains
internal and substitutional disorder outside this model; how far below is a
result. A geometry-only baseline (correlation of deposited $B$ against
$|\mathbf r-\mathbf r_{\rm cm}|^2$) is printed alongside, because a rigid body
reproduces real $B$ at $r\approx0.4$–$0.6$ from the lever arm alone. $R$ and $CC$
are reported per resolution shell, so the resolution at which the model stops
working is measured.

**Not yet included.** An additive non-lattice background. The deposited map still
contains solvent scattering and short-range internal motion, which a
single-rigid-body BvK model cannot represent, so the fit must absorb that mismatch
by distorting $K$. Adding a smooth $b(|q|)\,|F(\mathbf q)|^2$ term — linear in its
coefficients, so profilable in closed form alongside $s$ — is the natural next
step once the rest is clean.
