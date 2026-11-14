# Region tolerance (`REGION_RTOL`)

How much slack `ClusterRegions.contains` allows when checking points against a
cluster's region, why that slack is measured against the domain diagonal rather
than the cluster's radius, and the one assumption it still relies on.

Code: `REGION_RTOL` and `ClusterRegions` in `src/sampling.py`,
`BinaryTreeForest.ancestor_planes` in `src/trees.py`.

## What the tolerance is for

A cluster's region is the domain box cut by one half-space per ancestor. The
gathered row `(a, β)` of ancestor `j` means

$$
a \cdot (x - \mu) + \beta \le 0,
\qquad a = \frac{\sigma w}{\lVert w\rVert},
\qquad \beta = \frac{\sigma\,\bigl(b + w\cdot(\mu - c)\bigr)}{\lVert w\rVert},
$$

where `(w, b, c)` is the ancestor's stored plane `w·(x − c) + b = 0`, `c` the
ancestor's centroid, `μ` the cluster's centroid and `σ = ±1` the side.

The planes themselves are exact: nothing shifts a face. The tolerance is only
the slack in a membership check:

$$
\text{inside} \iff a\cdot(x-\mu) + \beta \le \varepsilon
\ \text{for every row, and}\ \ \text{lower} - \varepsilon \le x \le \text{upper} + \varepsilon,
$$

with

$$
\varepsilon = \texttt{REGION\_RTOL}\cdot D = 10^{-12}\cdot\lVert \text{upper} - \text{lower}\rVert .
$$

Cuts sit halfway between neighbouring projection scores, so points normally
lie strictly inside every face. A point lands on a face only when two scores
tie at the cut. Without slack, rounding could then put a valid point outside
its own cluster, and a test or sampler would reject it.

## Notation

- $u = 2^{-53} \approx 1.1\times10^{-16}$: unit roundoff of float64.
- $\gamma_n = nu/(1-nu)$: the standard constant in floating-point error bounds
  for an n-term dot product (Higham, *Accuracy and Stability of Numerical
  Algorithms*, §3.1).
- $\mathrm{fl}(\cdot)$: the computed value.
- $D$: the domain diagonal.

All bounds below are to first order in $u$.

## Result 1: rounding when re-expressing the plane is bounded by the domain (proved)

Write the exact value of the stored plane at a point as
$g(x) = \sigma\,(w\cdot(x-c) + b)/\lVert w\rVert$. Algebraically
$a\cdot(x-\mu) + \beta = g(x)$, so the computed value
$\hat v = \mathrm{fl}(\hat a\cdot(x-\mu) + \hat\beta)$ differs from $g(x)$ only
by rounding. Three sources contribute.

1. **The offset $\hat\beta$.** The numerator $b + \sum_i w_i(\mu_i - c_i)$ is a
   dot product of length $d$ plus one addition:
   $$
   \bigl|\mathrm{fl}(N) - N\bigr| \le \gamma_{d+2}\Bigl(|b| + \sum_i |w_i|\,|\mu_i - c_i|\Bigr)
   \le \gamma_{d+2}\bigl(|b| + \lVert w\rVert\,\lVert\mu - c\rVert\bigr),
   $$
   by Cauchy–Schwarz. Dividing by the computed $\lVert w\rVert$ adds a relative
   error of order $\gamma_{d+1}$. Together:
   $$
   |\hat\beta - \beta| \lesssim \gamma_{2d+4}\left(\frac{|b|}{\lVert w\rVert} + \lVert\mu - c\rVert\right).
   $$
2. **The term $\hat a\cdot(x-\mu)$.** $\hat a$ is a unit vector with
   componentwise relative error at most $\gamma_{d+2}$, and the dot product with
   $x - \mu$ adds $\gamma_d$:
   $$
   \bigl|\mathrm{fl}(\hat a\cdot(x-\mu)) - a\cdot(x-\mu)\bigr| \le \gamma_{2d+3}\,\lVert x-\mu\rVert .
   $$
3. **The final addition** adds at most $u\,(|a\cdot(x-\mu)| + |\beta|)$, and
   $|\beta| \le |b|/\lVert w\rVert + \lVert\mu - c\rVert$.

Summing:

$$
\boxed{\;|\hat v - g(x)| \;\le\; \gamma_{2d+6}\left(\frac{|b|}{\lVert w\rVert} + \lVert\mu - c\rVert + \lVert x - \mu\rVert\right)\;}
$$

**Each term is at most the domain diagonal $D$.** Every training point lies in
the domain box: training checks this for a given box, and the data's bounding
box contains the points by construction. Augmented points stay inside the box:
convex combinations stay in the cluster's hull, and a region-aware sampler
samples inside the region, which lies in the box. Every centroid is a mean of
such points, so it lies in the box too. Hence:

- $\lVert\mu - c\rVert \le D$ and $\lVert x - \mu\rVert \le D$.
- $|b|/\lVert w\rVert \le D$. The offset is $b = -t$, where $t$ is the midpoint
  of two of the ancestor's own projection scores $w\cdot(x_k - c)$, so
  $|t| \le \lVert w\rVert \max_k \lVert x_k - c\rVert$.

Therefore

$$
|\hat v - g(x)| \;\le\; 3\,\gamma_{2d+6}\,D \;\approx\; 3\,(2d+6)\,u\,D .
$$

This is at most $\varepsilon = 10^{-12} D$ whenever $3(2d+6)u \le 10^{-12}$,
which holds for **$d \lesssim 1{,}500$**. This is a worst-case bound; typical
rounding errors grow like $\sqrt d$ rather than $d$.

## Result 2: a tolerance proportional to the cluster's radius cannot work (proved)

The bound contains $\lVert\mu - c\rVert$ and $|b|/\lVert w\rVert$, which are set
by the *ancestor*, not by the cluster's radius $r = \max\lVert x - \mu\rVert$. A
deep cluster resting against an old ancestor's face is far from that ancestor's
centroid, and the ratio $\lVert\mu - c\rVert / r$ grows with every split below
the ancestor. So no fixed multiple of $u\,r$ bounds the error, and
$\varepsilon = 10^{-12}\,r$ (the value first agreed) is smaller than the
rounding once $\lVert\mu - c\rVert / r$ exceeds a few thousand.

## Open assumption: the split and the stored plane must agree

Result 1 bounds the error between the computed constraint and the **stored**
plane. But the split assigns points to sides with a different computation of
the same plane:

- the scores are $\hat s = \mathrm{fl}\bigl(v^\top \mathrm{fl}(M(x - c))\bigr)$
  (`run_fused_split_projection` in `src/numerics.py`);
- the stored normal is $\hat w = \mathrm{fl}(M^\top v)$ (`SplitStrategy.split`
  in `src/splitting.py`).

Both equal $w\cdot(x - c)$ in exact arithmetic, but their rounding differs by up to

$$
2\,\gamma_{d+1}\,\kappa\,\lVert x - c\rVert,
\qquad
\kappa = \frac{\bigl\lVert\, |M|^\top |v| \,\bigr\rVert}{\lVert M^\top v\rVert} \ \ge 1 .
$$

$\kappa$ measures cancellation in $M^\top v$. It is about 1 for a
well-conditioned metric and large when $v$ lies along directions that $M$
shrinks, which an anisotropic Hessian metric can produce. A point that tied at
the cut is at 0 in the split's arithmetic, but can be up to that distance on the
wrong side of the stored plane. $\varepsilon$ covers this only if

$$
\kappa\,(d + 1) \;\lesssim\; 4{,}500 \qquad (\text{i.e. } 10^{-12}/(2u)),
$$

and **nothing currently guarantees that**. No violation has been observed (see
below), but that is evidence, not proof.

### How to close the gap

1. **Score with the stored normal (recommended).** Compute the split scores as
   $\mathrm{fl}(\hat w\cdot(x - c))$ instead of $v^\top M(x - c)$, in both the
   NumPy path and the CuPy kernel. Membership and the stored plane then come
   from the same arithmetic, and Result 1 covers everything. This changes the
   scores at rounding level, so the CPU regression check will no longer be
   bit-identical. Per `AGENTS.md` it is a deliberate numerical change, to be
   listed and justified separately.
2. **Or measure $\kappa$** on real runs with the Hessian metrics, and state the
   assumption $\kappa(d+1) \lesssim 4{,}500$ next to `REGION_RTOL`.

## Measurements

These numbers support the analysis; they do not replace it.

**Method.** We trained from a single root with `track_trees=True`, work stealing
off, 20,000 points uniform in $[-2, 2]^d$ and `min_pts_per_cluster = 20`. At each
augmentation we took up to 40 rejected clusters per batch and up to 30 points
per cluster. For each (cluster, ancestor) pair we compared the constraint as
evaluated in float64 from the gathered rows with the stored plane evaluated
exactly in rational arithmetic (Python `fractions`).

| Run | Pairs | Deepest depth gap | Largest $\lVert\mu-c\rVert / r$ | Pairs with error $> 10^{-12} r$ | Largest error $/ D$ |
|---|---|---|---|---|---|
| 1-D, $f = \tanh(60x)$ | 3,985 | 20 | 8,934 | **54** (up to $2.8\times10^{-12} r$; depth gaps 11–19, ratio $\ge$ 2,355) | $1.1\times10^{-16}$ |
| 2-D, Rosenbrock | 3,735 | 17 | 161 | 0 | $1.6\times10^{-16}$ |

- The domain-relative error stays at a few $u$, more than 6,000× below
  $10^{-12}$, as Result 1 predicts.
- The radius-relative error grows with $\lVert\mu - c\rVert / r$ and crosses
  $10^{-12}$ in the 1-D run, as Result 2 predicts. The ratio climbs fastest in
  1-D, where every split cuts the same axis. The first failures appear near a
  ratio of 2,400 rather than $10^{-12}/u \approx 4{,}500$ because individual
  errors are a few units in the last place, not one.
- Separately, a 5-D Rosenbrock run (60,000 points, depth up to 28) checked
  5.3 million existing points against their gathered planes with **zero**
  slack, and none was outside. Every point sat at least about $10^{-9}$ (relative)
  inside its faces. This is consistent with the open assumption holding in
  practice, but it does not prove it.

## When the tolerance would be too loose

$\varepsilon$ accepts points up to $10^{-12} D$ outside a face. That only
matters for a cluster narrower than about $10^{-12}$ of the domain. Float64
coordinates at the domain's scale are spaced about $u\,D$ apart, so such a
cluster would barely be representable, and it could not hold
`min_pts_per_cluster` distinct points spread across it.
