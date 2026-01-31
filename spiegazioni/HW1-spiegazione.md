# Homework 1

## Exercise 1


In this exercise, we applied the Gradient Descent (GD) algorithm to the strictly convex function $\mathcal{L}(\theta) = (\theta - 3)^2 + 1$. We tested three distinct step sizes (learning rates), denoted by $\eta$, to observe their impact on the optimization trajectory.

### 1. Analysis of Convergence Behavior

The simulation results highlight three distinct regimes of optimization behavior based on the choice of $\eta$:

- **Small Step Size ($\eta = 0.05$): Slow Convergence**

    shows a very smooth, monotonic trajectory. The algorithm successfully minimizes the loss, moving from the initial guess towards the optimal value $\theta^\* = 3$. However, the convergence is **slow**. The updates are tiny, requiring a large number of iterations to traverse the curve. While this approach is safe (low risk of overshooting), it is computationally inefficient.
- **Medium Step Size ($\eta = 0.2$): Fast Convergence**

    demonstrates the "sweet spot" for this specific function. The steps are large enough to cover ground quickly but small enough to remain stable. The algorithm reaches the neighborhood of the minimum in very few iterations compared to the $\eta=0.05$ case. The loss curve drops sharply and stabilizes near 1, indicating efficient convergence.
- **Large Step Size ($\eta = 1.0$): Oscillation**

    illustrates a failure to converge. With $\eta = 1.0$, the algorithm enters a state of permanent **oscillation**. Instead of descending into the valley, the parameter bounces back and forth between $\theta = 0$ and $\theta = 6$. Consequently, the loss value remains constant at 10 (as seen in the Loss vs. Iteration graph) and never reaches the minimum. This is a classic example of the step size being too aggressive relative to the curvature of the function.

### 2. Theoretical Connections

The observed behaviors directly correlate with the theoretical properties of Gradient Descent discussed in class:

#### The "Goldilocks" Principle of Step Size

The results validate that the learning rate is a critical hyperparameter.

- **Too Small:** Convergence is guaranteed for convex functions, but it takes too long.
- **Too Large:** The updates overshoot the minimum. If the step is excessively large, the algorithm may diverge (go to infinity) or oscillate indefinitely (as seen with $\eta=1.0$).
- **Just Right:** A tuned step size ($\eta=0.2$) balances speed and stability, leveraging the gradient information effectively to minimize the objective function rapidly.

#### The Role of Convexity

The function $\mathcal{L}(\theta)$ is strictly convex (a quadratic bowl). Theoretically, convexity ensures that:

1. There is a unique global minimum ($\theta^\* = 3$).
2. The gradient always points towards this minimum.

However, **convexity alone guarantees convergence only if the step size is chosen correctly.** As seen in the $\eta=1.0$ case, even on a perfectly convex surface, a bad choice of step size can prevent the algorithm from ever reaching the bottom of the bowl.

#### Stability Bounds

For a quadratic function of the form $ax^2$, the maximum stable step size is related to the Lipschitz constant of the gradient (which is determined by the curvature). In this case, $\eta=1.0$ sits exactly on the boundary of stability, leading to pure oscillation. A step size slightly larger than $1.0$ would likely cause the algorithm to **diverge** (explode towards infinity), while anything smaller than $1.0$ would eventually converge.

## Exercise 2
### Discussion of Results

Based on the implemented **Gradient Descent with Backtracking** and the resulting plots, here is the analysis of the algorithm's behavior on the non-convex function $\mathcal{L}(\theta) = \theta^4 - 3\theta^2 + 2$.

#### 1. Why different initializations converge to different minima
Gradient Descent is a **local optimization algorithm**. It determines the direction of the update based solely on the gradient at the current point, without knowledge of the global landscape. The function used here is **non-convex**, possessing two distinct local minima (at $\theta \approx \pm 1.22$) and a local maximum at $\theta = 0$.

*   **Basins of Attraction:** The landscape is divided into "basins of attraction" separated by the local maximum (the "hill" at $\theta=0$).
*   **Initialization at $\theta_0 = -2.0$:** The point starts on the left side of the hill. The negative gradient pulls it further to the right, but it stays within the left basin, eventually converging to the **left local minimum**.
*   **Initialization at $\theta_0 = 2.0$:** Similarly, this point starts in the right basin and converges to the **right local minimum**.
*   **Initialization at $\theta_0 = 0.5$:** This point is crucial. Although it is numerically close to 0, it lies on the positive slope of the local maximum ($\theta > 0$). The gradient points strictly towards the right valley. Consequently, the algorithm treats it as belonging to the right basin of attraction and converges to the **right local minimum**, same as $\theta_0 = 2.0$.

#### 2. How backtracking automatically chooses a suitable step size
The polynomial $\theta^4$ creates a landscape with varying curvature: it is very flat near the minima but becomes extremely steep as $|\theta|$ increases.

*   **Mechanism:** At each iteration, the backtracking algorithm proposes a large step (initial $\eta = 1.0$). It checks the **Armijo condition**: does this step decrease the loss sufficiently relative to the gradient's magnitude?
*   **Observation:** In the "Loss vs. Iterations" plots, we see a monotonic decrease in loss without oscillations. This indicates that when the algorithm was at steep areas (e.g., $\theta = -2$), the Armijo condition failed for $\eta=1.0$, forcing the inner loop to multiply $\eta$ by $\beta$ (shrinking it) until the step was safe.
*   **Adaptivity:** As the parameters approached the flat valley of the minimum, the gradient magnitude decreased. The backtracking line search likely accepted larger step sizes (closer to the initial $\eta$) because the risk of overshooting was lower, allowing for precise convergence.

#### 3. Situations where constant step size would fail
A constant step size approach is highly sensitive to the scale of the gradient, which poses a major problem for this specific function.

*   **The Steepness Problem:** The gradient is $\nabla \mathcal{L} = 4\theta^3 - 6\theta$. At $\theta = -2$, the gradient is $-20$. At $\theta = -3$, it is $-90$. The magnitude grows cubically.
*   **Divergence:** If we chose a constant step size suitable for the detailed convergence near the minimum (e.g., $\eta = 0.1$), applied at $\theta = -3$, the update would be roughly $\Delta \theta \approx 0.1 \times 90 = 9.0$. This massive jump would send the parameter to the opposite side of the graph, likely even further out where the gradient is steeper, leading to numerical overflow (divergence).
*   **Oscillation:** If we chose a very small constant step size to prevent divergence at the edges (e.g., $\eta = 0.001$), the convergence near the flat minima would become excruciatingly slow, requiring thousands of iterations instead of the $\approx 5$ iterations achieved here with backtracking.
*   

## Exercise 3

### Code Explanation

The Python script implements the standard Gradient Descent (GD) algorithm with a constant step size to minimize a simple quadratic function $\mathcal{L}(\Theta) = \frac{1}{2}\Theta^T A \Theta$.

*   **Loss Function ($\mathcal{L}$):** Defined by the diagonal matrix $A = \text{diag}(1, 25)$. This structure means the loss is $\mathcal{L}(\theta_1, \theta_2) = 0.5 \cdot (1\cdot\theta_1^2 + 25\cdot\theta_2^2)$.
*   **Gradient Function ($\nabla\mathcal{L}$):** Calculated as $A\Theta$, specifically $[\theta_1, 25\theta_2]^T$.
*   **GD Function:** Performs the iterative update $\Theta^{(k+1)} = \Theta^{(k)} - \eta \nabla\mathcal{L}(\Theta^{(k)})$.
*   **Experiment Setup:** The code tests three different constant learning rates ($\eta = 0.02, 0.05, 0.1$) starting from the same point, $\Theta_0 = [5.0, 1.0]$.
*   **Visualization:** The resulting plots show the **level sets** (contours) of the ill-conditioned loss function overlaid with the path (trajectory) taken by the GD algorithm for each learning rate.

---

### Analysis of Results: Ill-Conditioning and Geometry

The plots perfectly illustrate the challenges Gradient Descent faces when optimizing an **ill-conditioned** objective function.

| Observation | $\eta = 0.02$ | $\eta = 0.05$ | $\eta = 0.1$ |
| :--- | :--- | :--- | :--- |
| **Path Behavior** | Slow but stable convergence. | Clear zig-zag and oscillation. | Rapid divergence/explosion. |

#### 1. The Elongated Ellipses Produced by Ill-Conditioning
The contour lines (level sets) are not circular but are highly **elongated** along the $\theta_1$ axis and compressed along the $\theta_2$ axis. This geometry is a direct result of the large **condition number** (ratio of the largest to smallest eigenvalues: $25/1 = 25$). The function is 25 times steeper in the $\theta_2$ direction than in the $\theta_1$ direction. This means movement along $\theta_1$ yields little change in loss, while movement along $\theta_2$ yields massive changes.

#### 2. The Gradient Direction Compared to the Level Set Lines
At any point on the trajectory (except near the origin), the gradient vector (which dictates the step direction) is always **perpendicular** to the contour line passing through that point. Because the contours are stretched, the perpendicular direction points severely toward the steepest edge ($\theta_2$), rather than pointing directly toward the target minimizer $(0, 0)$. This forces the algorithm to spend its initial steps correcting the steep dimension ($\theta_2$), rather than making progress toward $\theta_1$.

#### 3. Zig-Zag Behaviour for Large Condition Numbers
This behavior is clearly visible in the middle plot ($\eta = 0.05$).
*   When the learning rate is too large relative to the curvature in the steep dimension ($\theta_2$), the step **overshoots** the minimum along that axis, landing on the opposite side of the valley.
*   The subsequent step corrects the overshoot, but again overshoots the minimum.
*   This results in the characteristic back-and-forth **zig-zag pattern**, wasting steps correcting the same error repeatedly, preventing smooth convergence down the valley floor.
*   For $\eta = 0.1$, the overshooting is so massive that the trajectory explodes outside the visible range (or oscillates between extremely large values).

#### 4. The Relation to Slow Convergence in Narrow Valleys
The plot for $\eta=0.02$ (left) demonstrates the trade-off.
*   The algorithm quickly reduces the error in the steep $\theta_2$ dimension (in the first few steps), successfully navigating the narrow valley width.
*   However, because $\eta$ must be kept small (to avoid zigzagging in $\theta_2$), the movement along the shallow $\theta_1$ direction becomes extremely slow. The trajectory spends the majority of its iterations inching horizontally along the $\theta_1$ axis, demonstrating the inefficiency of GD in directions with low curvature. The convergence speed is dominated by the weakest (shallowest) direction.
*   

## Exercise 4

The plots exhibit the expected theoretical behavior for a quadratic problem:
1.  **Left Plot:** The **Exact Line Search (Blue)** shows the characteristic "orthogonal" path where every turning point is tangent to a level set (the new gradient is orthogonal to the previous search direction). The **Backtracking (Red)** follows a similar zig-zag path but with slightly different step lengths determined by the Armijo condition.
2.  **Right Plot:** Both methods show **linear convergence** (which appears as a straight line on a semi-log scale), which is the standard behavior for Gradient Descent on strongly convex functions.

### Brief Code Description
The code solves the optimization problem for the quadratic loss $\mathcal{L}(\theta) = \frac{1}{2}\theta^T A \theta$ where $A = \text{diag}(5, 2)$.
*   **Exact Line Search:** Calculates the mathematically optimal step size $\eta_k$ at each iteration using the formula $\eta_k = \frac{g_k^T g_k}{g_k^T A g_k}$. This ensures the maximum possible drop in loss along the current gradient direction.
*   **Backtracking:** Uses a heuristic loop. It starts with a large $\eta$ (e.g., 1.0) and iteratively shrinks it by a factor $\beta$ until the **Armijo condition** is met (sufficient decrease in loss).
*   **Visualization:** It compares the two methods by plotting their spatial trajectories over the contour map and their loss decay over time.

---

### Analysis of Results

*   **Speed of Convergence:**
    *   **Early Stages:** The **Exact Line Search (Blue)** is generally faster in the first few iterations (as seen in the trajectory plot, it gets to the center "valley" very quickly). This is because it computes the greedy optimal step.
    *   **Long Term:** Interestingly, in your log-plot, **Backtracking (Red)** actually achieves a lower loss in later iterations (steeper slope). This is a known phenomenon: while Exact Line Search is locally optimal (greedy), it forces the algorithm into a rigid "canonical zig-zag" pattern determined entirely by the condition number of matrix $A$. Backtracking, by being slightly "sub-optimal" and noisy, can sometimes accidentally step into a more favorable position that breaks the worst-case zig-zag cycle, allowing for slightly faster asymptotic convergence in specific setups.

*   **Smoothness of Step-sizes:**
    *   **Exact Line Search:** The step sizes are **mathematically determined** and "smooth" in the sense that they follow a strict geometric rule (steps are always orthogonal to the next gradient). The trajectory looks like a clean, rigid geometric pattern.
    *   **Backtracking:** The step sizes are **discrete and adaptive**. You might notice the red path takes steps that are sometimes larger or smaller in a less predictable pattern than the blue path. This is because the step size $\eta$ depends on how many times the `while` loop runs to satisfy the Armijo condition. If the condition is met immediately with $\eta=1$, a large step is taken; if not, the step is drastically cut.

## Exercise 5

### Code Description (New Parts)

Compared to previous exercises involving quadratic forms, this code introduces three key changes:

1.  **Non-Convex Objective Function:** The code implements the **Rosenbrock function** (`rosenbrock_loss`) and its manually derived gradient (`rosenbrock_grad`). Unlike the previous matrix-based quadratic functions ($x^T Ax$), this function includes non-linear terms (squares of squares), creating a complex, non-convex landscape with a curved "banana-shaped" valley.
2.  **Logarithmic Visualization:** Because the Rosenbrock function creates extremely high loss values away from the valley but very low values near the minimum, the code uses `np.logspace` for contour levels and sets the loss-vs-iteration y-axis to a logarithmic scale (`ax2.set_yscale('log')`) to visualize convergence meaningfuly.
3.  **Multiple Constants Comparison:** The execution loop now tests three specific small constant step sizes ($\eta \in \{10^{-3}, 10^{-4}, 10^{-5}\}$) alongside Backtracking for every starting point to highlight the sensitivity of hyperparameters.

---

### Analysis of Results

Based on the generated plots, here is the discussion regarding the behavior of Gradient Descent on the Rosenbrock landscape:

#### 1. Whether the method enters the valley
**Yes, all methods successfully enter the valley.**
In all four starting scenarios, even the smallest step sizes eventually descend from the high steep sides into the parabolic valley ($\theta_2 \approx \theta_1^2$). This is because the gradient magnitude on the "walls" of the valley is massive, forcing the algorithm quickly toward the floor of the valley regardless of the specific step size (provided it doesn't diverge).

#### 2. How long it takes to “turn” correctly along the valley direction
**It takes a significant amount of iterations, and constant step sizes struggle immensely here.**
Once inside the valley, the gradient points mostly toward the steep walls rather than along the gentle slope toward the global minimum $(1, 1)$.
*   **Constant Step Sizes:** Methods with small $\eta$ (e.g., $10^{-4}, 10^{-5}$) get "stuck" bouncing slightly or moving infinitesimally slowly along the valley floor. They struggle to align the descent direction with the curvature of the banana shape.
*   **Backtracking:** It adapts much faster. For example, in the `Start: [-1.5, 2.0]` plot, the red line enters the valley and makes the sharp right turn toward $(1, 1)$ much earlier than the constant step approaches.

#### 3. Whether the method zig-zags
**Yes, zig-zagging is prevalent, especially for constant step sizes.**
The Rosenbrock valley acts like a narrow canyon.
*   **Constant $\eta$:** If the step size is slightly too large (e.g., $\eta=10^{-3}$ in some regions), the trajectory bounces back and forth between the valley walls (oscillating) rather than moving down the center. This wastes computational effort.
*   **Backtracking:** While it also exhibits some zig-zagging when entering the valley (where gradients change rapidly), the adaptive line search helps dampen these oscillations by reducing the step size exactly when the curvature forces a turn, allowing for a smoother path compared to the aggressive constant steps.

#### 4. Whether step sizes become too small or too large
*   **Too Small (Constant $\eta$):** The values $\eta=10^{-4}$ and $\eta=10^{-5}$ (Orange and Blue lines) are far too small for this problem. While safe, their progress is agonizingly slow. In the log-plots, they appear nearly flat compared to backtracking, indicating they would require millions of iterations to reach the minimum.
*   **Too Large / Divergence:** While explicit divergence isn't visually dominated in these specific plots (due to the small constants chosen), $\eta=10^{-3}$ (Cyan) often shows instability or slower convergence than backtracking because it is too coarse for the intricate valley floor.
*   **Backtracking Efficiency:** Backtracking strikes the balance. It utilizes large steps when far from the valley (descending quickly) and automatically shrinks the steps to navigate the narrow, curved valley floor without diverging, achieving the lowest loss in all test cases.