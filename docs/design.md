# Design notes

The derivations behind each module, and the reason for every choice that is not forced by the
papers. Equation numbers refer to Kerbl et al. 2023 unless stated otherwise. Measured values
are in the README and in `results/`; this file only contains mathematics and decisions.

## 1. Parameterisation (`gaussians.py`)

A Gaussian is $(\mu, \Sigma, o, c)$. Gradient descent needs unconstrained variables, so each
constrained quantity is the image of a free one:

| Quantity | Constraint | Free variable | Map |
|:--|:--|:--|:--|
| scale $s$ | $s > 0$ | `log_scales` | $s = e^{\ell}$ |
| rotation $R$ | $R \in SO(3)$ | `quats` (4 numbers) | $R(q / \lVert q \rVert)$ |
| opacity $o$ | $0 < o < 1$ | `opacity_logits` | $o = 1/(1+e^{-x})$ |
| colour $c$ | $c \ge 0$ | `sh_dc` | $c = \max(0,\ 0.5 + Y_0^0 f)$, $Y_0^0 = 1/(2\sqrt{\pi})$ |

The covariance is built as $\Sigma = R S S^\top R^\top$ with $S = \mathrm{diag}(s)$ (eq. 6). It is
symmetric positive definite for any value of the free variables, which a direct optimisation
of the six entries of $\Sigma$ would not guarantee. Its eigenvalues are the $s_k^2$ and it does
not depend on $\lVert q \rVert$; both properties are tested.

The colour keeps the spherical-harmonic encoding even though only degree 0 is used, because
that is what the `.ply` exchange format stores.

## 2. Camera (`camera.py`)

OpenCV convention: $t = W p + T$, the camera looks along $+z$, $x$ right, $y$ down, and

$$u = f_x \frac{t_x}{t_z} + c_x, \qquad v = f_y \frac{t_y}{t_z} + c_y .$$

Pixel $(i, j)$ covers $[i, i+1) \times [j, j+1)$ and is sampled at its centre
$(i + 0.5,\ j + 0.5)$. The ray tracer that produces the ground truth uses the same convention;
a test places a small Gaussian at the centre of a ray-traced sphere and checks that both
codes put it at the same sub-pixel position.

## 3. Projection of a Gaussian (`reference.project_gaussians`)

A Gaussian pushed through a perspective projection $\pi$ is no longer a Gaussian. EWA
splatting (Zwicker et al. 2002) replaces $\pi$ by its first-order expansion at the centre
$t_0 = W\mu + T$:

$$\pi(t) \approx \pi(t_0) + J\,(t - t_0), \qquad
J = \frac{\partial \pi}{\partial t}\Big|_{t_0} =
\begin{pmatrix} f_x/t_z & 0 & -f_x t_x / t_z^2 \\ 0 & f_y/t_z & -f_y t_y / t_z^2 \end{pmatrix}.$$

The image of $\mathcal N(\mu, \Sigma)$ under an affine map $p \mapsto A p + b$ is
$\mathcal N(A\mu + b,\ A \Sigma A^\top)$. With $A = J W$ this gives eq. 5:

$$\mu' = \pi(W\mu + T), \qquad \Sigma' = J W \Sigma W^\top J^\top .$$

The neglected terms are of second order in (size of the Gaussian) / (depth). Two tests bracket
this: $J$ is compared with central differences of $\pi$, and $\Sigma'$ is compared with the
sample covariance of 3D samples projected exactly. The second test is run for a small
Gaussian, where it must agree, and for a large one, where it must not.

**Low-pass term.** $0.3\,I$ (pixel²) is added to $\Sigma'$. Adding covariances is convolving
with a Gaussian of standard deviation $\sqrt{0.3} \approx 0.55$ pixel, i.e. a crude integration
over the pixel area. Without it a Gaussian narrower than a pixel can fall between pixel centres,
contribute nothing and receive no gradient. The peak opacity is not rescaled after the
dilation, so very small Gaussians are rendered slightly too bright; Mip-Splatting (Yu et al.
2024) corrects this and it is listed as a limitation here.

## 4. Compositing (`reference.render`)

Emission-absorption volume rendering along a ray gives $C = \int T(s)\,\sigma(s)\,c(s)\,ds$
with $T(s) = \exp(-\int_0^s \sigma)$. Cut into ordered slabs of opacity $\alpha_i$, it becomes
eq. 3:

$$C = \sum_i c_i\,\alpha_i\,T_i + T_{\text{end}}\,c_{\text{bg}}, \qquad
T_i = \prod_{j<i} (1 - \alpha_j).$$

3DGS takes each Gaussian as one slab located at the depth of its centre, with

$$\alpha_i(x) = \min\!\big(0.99,\ o_i \exp(-\tfrac12 (x-\mu_i')^\top \Sigma_i'^{-1} (x-\mu_i'))\big),$$

set to zero below $1/255$. Consequences that the tests pin:

* $w_i = \alpha_i T_i = T_i - T_{i+1}$ telescopes, so $\sum_i w_i + T_{\text{end}} = 1$: a scene
  whose Gaussians all have the background colour renders as a constant image.
* The accumulated opacity $1 - T_{\text{end}}$ does not depend on the order, the colour does.
* The order is the depth of the *centre*, one order per image. Two overlapping Gaussians may
  therefore be composited in an order that is wrong for some rays (the "popping" artefact of
  the method); this implementation shares it.
* The clamp at 0.99 keeps $1 - \alpha$ away from zero, so something always remains visible
  behind an opaque splat and the transmittance product has no zero factor.

The depth buffer is $\sum_i w_i z_i$. Dividing it by the accumulated opacity gives the expected
depth of the ray, which is what the depth-error metric uses.

## 5. Exact footprints and tiles (`torch_renderer.py`)

The threshold gives every Gaussian compact support:

$$o\,e^{-\frac12 d^\top \Sigma'^{-1} d} \ge \alpha_{\min}
\iff d^\top \Sigma'^{-1} d \le k, \qquad k = 2 \ln (o / \alpha_{\min}).$$

The largest $d_x$ on that ellipse is $\sqrt{k\,\Sigma'_{xx}}$ (support function of an ellipse;
one Lagrange multiplier), and likewise for $d_y$. The axis-aligned box of half-sizes
$\sqrt{k\,\Sigma'_{xx}}, \sqrt{k\,\Sigma'_{yy}}$ therefore contains every pixel the Gaussian can
touch, and nothing is lost by skipping the tiles outside it. This is why the tiled twin can be
required to equal the un-culled reference to rounding.

The reference implementation bounds a splat by 3 standard deviations of its major axis
instead. At $3\sigma$ the opacity is $o\,e^{-4.5} \approx 0.011\,o$, above $1/255$ as soon as
$o > 0.35$, so that rule clips a faint but non-zero rim. It is a sensible speed trade-off, but
it would make a parity test against a dense renderer fail by construction, hence the exact
rule here. A 1 % margin on the box absorbs rounding. A Gaussian with $o \le \alpha_{\min}$ has
$k \le 0$ and is culled everywhere.

Evaluation per tile: the Gaussians overlapping it, in global depth order, form a padded
$(\text{tiles}, K)$ index table, and all quantities are tensors of shape
$(\text{tiles}, K, S^2)$ with $S = 16$. Two details keep this affordable on a CPU:

* *Chunks.* Tiles are sorted by their number of Gaussians and processed in groups whose padded
  tensor holds at most $2^{19}$ values (2 MB in float32). Peak memory is bounded by that budget
  times the number of live temporaries, not by the scene.
* *Separable exponent.* $-\tfrac12 d^\top Q d = -\tfrac12 q_a d_x^2 - q_b d_x d_y - \tfrac12 q_c d_y^2$.
  The two squares only depend on the column or on the row of the pixel, so they are computed on
  $(\text{tiles}, K, S)$ tensors; only the cross term needs the full grid.

Tile size and chunk budget are free parameters; tests check that the image does not depend on
either.

## 6. What autograd differentiates

Every step above is a differentiable tensor expression, so no backward pass is written by
hand. The render is only *piecewise* smooth in the parameters:

| Source | Effect |
|:--|:--|
| threshold $\alpha < 1/255 \to 0$ | jump of at most $\alpha_{\min}$ in one pixel, zero gradient outside the footprint |
| clamp at 0.99, colour clamp at 0 | kinks; zero gradient on the clamped side |
| depth order | a jump when two overlapping Gaussians swap |
| near plane | a Gaussian disappears when its centre crosses it |

Autograd returns the gradient of the smooth piece the parameters are in. The gradient test is
therefore run twice: with $\alpha_{\min} = 0$, where the function is smooth and central
differences must agree, and with the default threshold, where they agree as long as no pixel
crosses the threshold inside the finite-difference step. `torch.autograd.gradcheck` is run on
the smooth version as an independent check.

**Float32 depth ties.** The parity check in float32 does not always reach $10^{-5}$: among
1,000 random centres, two can be closer in depth than float32 resolves. The float32 twin then
sorts them in the other order and the pixels they share change colour. The check reports
whether both renderers used the same order, and a test asserts the signature of this effect:
colour differs, accumulated opacity does not.

## 7. Optimisation (`fit.py`, `optim.py`, `density.py`)

Loss (eq. 7): $(1-\lambda)\,L_1 + \lambda\,(1 - \mathrm{SSIM})$ with $\lambda = 0.2$, one random
training view per iteration, views drawn without replacement within an epoch.

| Group | Paper (30,000 it.) | Here (3,000 it.) |
|:--|--:|--:|
| positions, times scene extent | 1.6e-4, decays to 1 % | 8e-4, decays to 2 % |
| log-scales | 0.005 | 0.01 |
| rotations | 0.001 | 0.002 |
| opacity logits | 0.05 | 0.05 |
| colours | 0.0025 | 0.01 |

The rates were raised once, by hand, because the schedule is ten times shorter; they were not
tuned per scene or per seed. The position rate is multiplied by the scene extent so that the
behaviour does not depend on the unit of length, and decays log-linearly: large early moves,
fine late ones.

**Adam** is written out (thirty lines) because densification must resize its state: moments of
surviving Gaussians are kept, new Gaussians start from zero. It is checked step by step against
`torch.optim.Adam`. Epsilon is $10^{-15}$ as in the reference implementation; per-Gaussian
gradients of a mean-reduced image loss are small, and the PyTorch default of $10^{-8}$ would
damp the updates of faint Gaussians.

**Density control** (section 5.2 of the paper). For each Gaussian the norm of
$\partial L / \partial \mu'$ is averaged over the views in which it was visible. Every 100
iterations between 200 and 2,000, a Gaussian above the threshold is cloned if its largest
standard deviation is below 1 % of the scene extent, and otherwise replaced by two children
drawn from its own density with scales divided by 1.6. Gaussians with opacity below 0.005 are
removed, and the total is capped at 3,000 (largest gradients first). The threshold, 4e-5, is
expressed for centres in pixels and a mean-reduced loss on 128 x 128 images; the paper's 0.0002
is for normalised coordinates and much larger images, so the two values are not comparable.

Not implemented: the periodic opacity reset and the pruning of very large Gaussians, both
aimed at long runs on real captures; spherical harmonics above degree 0, since both test
scenes are view-independent.

## 8. Ground truth and metrics (`scenes.py`, `metrics.py`)

* `shapes` is ray traced (spheres, checkered disc, Lambertian shading, hard shadows, 16 rays
  per pixel). No Gaussian set reproduces its edges exactly, so the PSNR measures the
  representation as well as the optimiser. Its depth is exact.
* `knot` is rendered by the NumPy reference from 160 hidden Gaussians. The model can represent
  it exactly, so the PSNR measures the optimiser alone, and the targets come from the *other*
  implementation than the one being optimised.
* Training cameras: two rings of 12 at 20° and 50° of elevation. Held-out cameras: 8 on a ring
  at 35°, at azimuths halfway between training azimuths.
* PSNR and SSIM are averaged over the held-out views. SSIM follows Wang et al. 2004 with an
  11 x 11 Gaussian window and no padding (windows that fit entirely inside the image); it is
  checked against a direct sliding-window implementation.
* Depth error: median over pixels of $|\hat z - z| / z$, where $\hat z$ is the alpha-normalised
  rendered depth, on pixels where the scene has a surface and the render is more than half
  opaque. It is reported next to the PSNR because a splat scene can match images well while
  its depth is off, which matters as soon as the scene is used for geometry.

## 9. What each test would catch

| Test | Bug it would reveal |
|:--|:--|
| Jacobian vs finite differences | wrong sign or power of $t_z$ in $J$ |
| $\Sigma'$ vs Monte-Carlo | $W$ and $J$ applied in the wrong order, transposed rotation |
| eigenvalues of $\Sigma$ | scale applied on the wrong side of $R$ |
| analytic single footprint | pixel-centre offset, wrong conic, wrong blending weights |
| permutation invariance | missing or unstable depth sort |
| partition of unity | background blended with the wrong transmittance |
| NumPy vs PyTorch, several tilings | culling that drops visible pixels, tile assembly errors |
| autograd vs finite differences | detached tensors, in-place operations breaking the graph |
| Adam vs `torch.optim.Adam` | bias correction or epsilon placement |
| density-control bookkeeping | optimiser moments attached to the wrong Gaussian after a resize |
| `.ply` round trip | property order, activated instead of raw parameters |

## 10. Differences from the reference implementation

* Culling box exact for the alpha threshold instead of $3\sigma$ (section 5).
* No early termination when the transmittance falls below $10^{-4}$: it is a speed
  optimisation, and it changes pixel values by less than $10^{-4}$.
* No clamping of $t_x / t_z$, $t_y / t_z$ in the Jacobian for Gaussians far outside the
  frustum; the test scenes keep everything in view.
* Degree-0 colour only, no opacity reset, no size-based pruning (section 7).
* Gradients from autograd rather than a hand-written backward pass.
* No code from that implementation was used; everything is written from the equations of the
  two papers.
