# ft_linear_regression

> A complete, step-by-step guide to the mandatory part — with math, diagrams, and every confusion answered.

---

## Table of Contents

1. [The Hypothesis — Our Prediction Line](#1-the-hypothesis--our-prediction-line)
2. [The Cost Function J(θ₀, θ₁)](#2-the-cost-function-jθ₀-θ₁)
3. [Why We Need Derivatives](#3-why-we-need-derivatives)
4. [Computing the Partial Derivatives](#4-computing-the-partial-derivatives)
5. [The Update Rule — Why the Minus Sign](#5-the-update-rule--why-the-minus-sign)
6. [Gradient Descent — The Full Algorithm](#6-gradient-descent--the-full-algorithm)
7. [Stopping Condition](#7-stopping-condition)
8. [After Training — Making Predictions](#8-after-training--making-predictions)
9. [Quick-Reference Summary](#9-quick-reference-summary)

---

## 1. The Hypothesis — Our Prediction Line

We assume that the price of a car depends **linearly** on its mileage. That means our model is a straight line:

```
hθ(x) = θ₀ + θ₁ · x
```

| Symbol | Meaning |
|--------|---------|
| `x` | mileage (input feature) |
| `hθ(x)` | predicted price (output) |
| `θ₀` (theta-zero) | intercept — predicted price when mileage = 0 |
| `θ₁` (theta-one) | slope — how much price changes per km |

The project specifies that you start with **θ₀ = 0** and **θ₁ = 0**. That means your first prediction for every car is price = 0. That's completely wrong — but it doesn't matter. The algorithm will fix it.

```
price
  ^
  |                              ● after training (green line)
  |                         ●
  |                    ●
  |               ●
  |          ●
  |     ●
  |__________________________________> mileage
  θ₀=0, θ₁=0 → flat line at 0 (starting guess)
```

---

## 2. The Cost Function J(θ₀, θ₁)

We need a single number that measures **how wrong our current line is**. We use the Mean Squared Error, divided by 2 for convenience:

```
J(θ₀, θ₁) = (1 / 2m) · Σᵢ (hθ(xⁱ) − yⁱ)²
```

| Symbol | Meaning |
|--------|---------|
| `m` | number of cars in the dataset |
| `xⁱ` | mileage of car i |
| `yⁱ` | actual price of car i |
| `hθ(xⁱ) − yⁱ` | error: predicted price minus actual price |

### Why each design choice?

| Choice | Why |
|--------|-----|
| We **square** the error | Prevents positive and negative errors from cancelling. Also punishes large errors more than small ones. |
| We **divide by m** | Gets the average error so J doesn't grow just because the dataset is larger. |
| We **divide by 2** | Pure convenience: the 2 cancels when we take the derivative, making the formula cleaner. It doesn't change *where* the minimum is. |

### ❓ Confusion: "How can we compute J if we don't have the real hθ?"

**hθ(x) is always computable.** It's not the "true" model — it's just our current guess: `θ₀ + θ₁ · x`. At any point in training, we know θ₀ and θ₁ (we initialized them to 0), and we know every `xⁱ` from the dataset. So we can always compute `hθ(xⁱ)` and therefore J.

The dataset (all `xⁱ` and `yⁱ`) is fixed. The only things that change are θ₀ and θ₁. J is just a function of those two parameters.

### ❓ Confusion: "Why write J(θ₀, θ₁) instead of just J?"

Because J depends **only** on θ₀ and θ₁. The data is fixed — we don't change it. Writing `J(θ₀, θ₁)` makes it explicit: the only knobs we control are the two parameters. Our goal is to find the (θ₀, θ₁) pair that makes J as small as possible.

---

## 3. Why We Need Derivatives

Think of `J(θ₀, θ₁)` as a landscape — a surface above a 2D plane of (θ₀, θ₁) values. The height of the surface at any point is the cost. We want to find the lowest valley.

```
J
^
|      *         *
|        *     *
|          * *
|           *   ← minimum (what we want)
|
+-----------------> θ₁ (same shape for θ₀)
```

To navigate downhill, we need to know the **slope of the surface** in each direction. That's what derivatives give us:

```
slope < 0                     slope > 0
(going down to the right)     (going up to the right)
→ move RIGHT (increase t)     → move LEFT (decrease t)
```

For a function of **two variables** (θ₀ and θ₁), we need **partial derivatives** — one for each direction:

| Partial derivative | Meaning |
|--------------------|---------|
| `∂J/∂θ₀` | How J changes when we nudge θ₀ (keeping θ₁ fixed) |
| `∂J/∂θ₁` | How J changes when we nudge θ₁ (keeping θ₀ fixed) |

The downhill direction is always **opposite to the sign** of each partial derivative.

---

## 4. Computing the Partial Derivatives

Using the chain rule on J, we get:

```
∂J/∂θ₀ = (1/m) · Σᵢ (hθ(xⁱ) − yⁱ)

∂J/∂θ₁ = (1/m) · Σᵢ (hθ(xⁱ) − yⁱ) · xⁱ
```

### ❓ Confusion: "Why are the two formulas different? Why does θ₁ have an extra xⁱ?"

Let's think about what each parameter controls:

- **θ₀** is the intercept. If you increase θ₀ by 1, *every* prediction increases by exactly 1, regardless of the mileage. So the effect on J depends only on the average prediction error.

- **θ₁** is the slope. If you increase θ₁ by 1, the prediction changes by `1 · xⁱ` — which is **different for each car**. A car with mileage 200,000 km is affected 200,000× more than a car with mileage 1 km. So the correction for θ₁ must be *weighted by the mileage* of each car. That's where the extra `xⁱ` comes from.

In short: **θ₀ shifts the whole line up/down uniformly. θ₁ rotates the line** — and the effect of rotation grows with distance from zero.

### Derivation sketch (for completeness)

```
J = (1/2m) · Σ (θ₀ + θ₁·x - y)²

∂J/∂θ₀: chain rule → 2·(θ₀ + θ₁·x - y) · ∂(θ₀ + θ₁·x)/∂θ₀
        = 2·error · 1          (∂θ₀/∂θ₀ = 1)
        divide by 2m → (1/m)·Σ error

∂J/∂θ₁: chain rule → 2·(θ₀ + θ₁·x - y) · ∂(θ₀ + θ₁·x)/∂θ₁
        = 2·error · x          (∂(θ₁·x)/∂θ₁ = x)
        divide by 2m → (1/m)·Σ error · x
```

> The 2 from the square cancels with the 2 in the denominator of J — that's the only reason we put it there.

---

## 5. The Update Rule — Why the Minus Sign

The core update formula is:

```
t_new = t_old − α · f′(t_old)
```

where `α > 0` is the **learning rate** (step size).

### The three cases

| Case | Derivative | Direction to go downhill | What the formula does |
|------|-----------|--------------------------|----------------------|
| Slope is **positive** | `f′(t) > 0` | Move **left** (decrease t) | `t − α·(positive)` = smaller t ✓ |
| Slope is **negative** | `f′(t) < 0` | Move **right** (increase t) | `t − α·(negative)` = larger t ✓ |
| Slope is **zero** | `f′(t) = 0` | Stay put (we're at the minimum) | `t − α·0` = t (no change) ✓ |

### ❓ The key insight that clears the blur

The minus sign encodes the rule **"move opposite to the slope"** in one elegant operation. You don't need to check the sign of the derivative and then decide which way to go — the formula does it automatically.

If you used **plus** instead: `t + α·f′(t)`, you'd always move *with* the slope — uphill. You'd never converge to a minimum.

### Concrete example with f(t) = t²

`f′(t) = 2t`, minimum at `t = 0`.

**At t = −3:**
```
f′(−3) = −6   (negative → we are to the left of the minimum, go right)
t_new = −3 − 0.1·(−6) = −3 + 0.6 = −2.4   ✓ moved right (closer to 0)
```

**At t = +3:**
```
f′(3) = 6    (positive → we are to the right of the minimum, go left)
t_new = 3 − 0.1·6 = 3 − 0.6 = 2.4         ✓ moved left (closer to 0)
```

**At t = 0 (minimum):**
```
f′(0) = 0
t_new = 0 − 0.1·0 = 0                      ✓ no change (we stay at the minimum)
```

### Memory trick

> Think of the derivative as telling you **which way is up**. You want to go down, so you go the **opposite way**. The formula `θ_new = θ − α · derivative` does exactly that.

---

## 6. Gradient Descent — The Full Algorithm

For our two-parameter case, the update rule is:

```
tmp₀ = θ₀ − α · (1/m) · Σ (hθ(xⁱ) − yⁱ)
tmp₁ = θ₁ − α · (1/m) · Σ (hθ(xⁱ) − yⁱ) · xⁱ

θ₀ := tmp₀
θ₁ := tmp₁
```

### ❓ Confusion: "Why compute tmp₀ and tmp₁ first? Why 'simultaneous update'?"

Both gradients must be computed using the **same old** θ₀ and θ₁. If you update θ₀ first and then use the *new* θ₀ to compute the gradient for θ₁, you're no longer computing the true gradient of J at the original point — you've drifted off the correct downhill direction.

The fix: compute both new values into temporary variables, then assign both at once. This guarantees you're always moving from the same point in parameter space.

### Pseudocode

```python
# Initialize
θ0 = 0.0
θ1 = 0.0
α  = 0.001   # learning rate — tune this

# Repeat until convergence
for iteration in range(num_iterations):
    errors = [θ0 + θ1 * x[i] - y[i] for i in range(m)]

    grad0 = (1/m) * sum(errors)
    grad1 = (1/m) * sum(errors[i] * x[i] for i in range(m))

    θ0 = θ0 - α * grad0    # simultaneous!
    θ1 = θ1 - α * grad1    # simultaneous!
```

### One iteration, step by step

```
Step 1  Compute predictions hθ(xⁱ) for all cars
        → use current θ₀ and θ₁ to predict prices

Step 2  Compute errors eⁱ = hθ(xⁱ) − yⁱ
        → positive error = we predicted too high
        → negative error = we predicted too low

Step 3  Compute both gradients (don't update yet!)
        → grad₀ = mean of errors
        → grad₁ = mean of (errors × mileages)

Step 4  Update θ₀ and θ₁ simultaneously
        → subtract α × gradient from each parameter

↺  Repeat until convergence
```

### Cost J decreasing over iterations

```
J
^
|*
| *
|  *
|   **
|     ***
|        *****
|              ***********
+---------------------------------> iteration
```

Each iteration, the line fits the data a little better and J gets a little smaller.

---

## 7. Stopping Condition

We stop when the parameters are no longer changing significantly.

| Method | When to use |
|--------|-------------|
| Fixed number of iterations | Simplest — just run for e.g. 10,000 iterations. Good enough for this project. |
| `\|J(new) − J(old)\| < ε` | Stop when the cost barely changes between steps. More principled. |
| Both gradients ≈ 0 | Stop when both `∂J/∂θ₀` and `∂J/∂θ₁` are near zero. True convergence check. |

### Why J is convex — there's only one minimum

For linear regression, `J(θ₀, θ₁)` is a **convex** function — its graph is a perfect bowl shape. There are no local minima, no saddle points, no traps. Gradient descent with a reasonable learning rate is **guaranteed** to find the global minimum.

### ❓ What if α (learning rate) is too large?

If α is too large, each step **overshoots** the minimum. Instead of converging, J oscillates or even grows.

```
α too large:                    α just right:
J ^                             J ^
  |  *       *                    |*
  |    *   *                      |  *
  |      *                        |    **
  |    *   *                      |      ***
  |  *       *                    |         *****
  +-----------> iter              +-----------> iter
  (diverges or oscillates)        (converges smoothly)
```

The rule of thumb: start with `α = 0.001` or `α = 0.01` and watch J decrease. If J goes up, reduce α. If it converges very slowly, carefully increase α.

---

## 8. After Training — Making Predictions

Once gradient descent has converged:

1. **Save θ₀ and θ₁** to a file (e.g. `thetas.csv` or `model.json`).
2. **In the prediction program**, load θ₀ and θ₁ from that file.
3. **For any input mileage x**, compute:

```
price = θ₀ + θ₁ · mileage
```

4. If no model has been trained yet, default to `θ₀ = 0, θ₁ = 0` → prediction is 0.

That's it. The mandatory part has exactly two programs:
- **Program 1 (training):** reads the dataset, runs gradient descent, saves θ₀ and θ₁.
- **Program 2 (prediction):** loads θ₀ and θ₁, asks for a mileage, outputs the estimated price.

---

## 9. Quick-Reference Summary

### All formulas in one place

```
Hypothesis:    hθ(x) = θ₀ + θ₁ · x

Cost:          J(θ₀, θ₁) = (1/2m) · Σ (hθ(xⁱ) − yⁱ)²

Gradient θ₀:  ∂J/∂θ₀ = (1/m) · Σ (hθ(xⁱ) − yⁱ)
Gradient θ₁:  ∂J/∂θ₁ = (1/m) · Σ (hθ(xⁱ) − yⁱ) · xⁱ

Update:        θ₀ := θ₀ − α · ∂J/∂θ₀
               θ₁ := θ₁ − α · ∂J/∂θ₁   (simultaneously!)
```

### Key confusions answered

| Question | One-line answer | Full answer |
|----------|----------------|-------------|
| How can we compute J without the real hθ? | hθ is just our current guess, always computable. | [Section 2](#2-the-cost-function-jθ₀-θ₁) |
| Why write J(θ₀, θ₁)? | J depends only on the parameters, not the fixed data. | [Section 2](#2-the-cost-function-jθ₀-θ₁) |
| Why different formulas for θ₀ and θ₁? | θ₁ affects predictions scaled by mileage → needs xⁱ weighting. | [Section 4](#4-computing-the-partial-derivatives) |
| Why the minus sign in the update? | It ensures we always move opposite to the slope → downhill. | [Section 5](#5-the-update-rule--why-the-minus-sign) |
| What if slope is already negative (downhill)? | minus × negative = plus → we increase t → still downhill. | [Section 5](#5-the-update-rule--why-the-minus-sign) |
| Why simultaneous update? | Both gradients must use the same old θ values. | [Section 6](#6-gradient-descent--the-full-algorithm) |
| How do we know we're at the minimum? | Both gradients are (near) zero. J stops decreasing. | [Section 7](#7-stopping-condition) |

---

*Written as part of the 42 School ft_linear_regression project.*