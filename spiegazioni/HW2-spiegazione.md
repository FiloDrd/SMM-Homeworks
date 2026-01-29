# Exercise 2.1

### Analysis of Gradient Descent Variants

#### 1. Why Full Gradient Descent (GD) is smooth but slow for large $N$
* **Smoothness (Deterministic Trajectory):** Full Batch Gradient Descent computes the gradient using the **entire dataset** ($N$) for every single update step. Because it calculates the exact average gradient over all data points, the resulting vector points directly toward the steepest descent of the global cost function. This results in a smooth, deterministic trajectory without fluctuations.
* **Slowness (Computational Redundancy):** The downside is computational cost. If $N$ is 1,000,000, the algorithm must calculate the error and gradient for one million points just to move the parameters ($\theta$) a single tiny step. This makes it extremely slow per iteration and computationally expensive, especially if the data does not fit into memory.

#### 2. Why Stochastic Gradient Descent (SGD) is noisy but progresses faster
* **Noise (High Variance):** SGD uses a **single random data point** (batch size = 1) to estimate the gradient. Since one data point is a very rough approximation of the whole dataset, the gradient estimate is "noisy." The algorithm might move in a direction that is optimal for that specific point but suboptimal for the dataset as a whole, causing the trajectory to "zigzag" or oscillate wildly.
* **Speed (Frequent Updates):** despite the zigzagging, SGD is often "faster" in terms of wall-clock time to reach a decent error rate. It updates the parameters $N$ times in a single epoch, whereas Full GD updates only once per epoch. This allows the model to learn immediately and continuously, often getting close to the minimum long before Full GD has finished its first few epochs.

#### 3. How batch size affects noise level and convergence stability
The batch size controls the trade-off between the accuracy of the gradient estimate and the speed of computation:
* **Small Batch (High Noise):** With a small batch size, the variance of the gradient estimate is high. This noise prevents the model from settling perfectly into the minimum (it tends to wander around it), requiring a decaying learning rate to force convergence. However, this noise can be beneficial as it adds a form of regularization and helps the model escape shallow local minima or saddle points.
* **Large Batch (High Stability):** As batch size increases, the gradient estimate approaches the true gradient (Full GD). The path becomes smoother and convergence is more stable, allowing for larger learning rates. However, if the batch is too large, you lose the computational benefits of stochastic updates and the "exploration" benefits of noise.
* **The "Sweet Spot":** Mini-batch GD (e.g., sizes 32, 64, 128) is usually preferred because it utilizes matrix vectorization (SIMD instructions) efficiently to speed up calculation while maintaining enough stochastic noise to generalize well.


# Exercise 2.2

### Discussion: Variance of the Stochastic Gradient

#### 1. Why the variance decreases with larger batches
* **Averaging out the Noise:** The data inherently contains noise (random fluctuations not representative of the true underlying pattern). When the batch size ($N_{batch}$) is small (e.g., 1), the gradient calculation is dominated by the noise of that single data point, leading to a high variance.
* **Law of Large Numbers:** As you increase the batch size, you are averaging the gradient over more data points. According to the Law of Large Numbers, the average of a sample converges to the expected value (the true full-batch gradient) as the sample size increases. The random "errors" (noise) in individual points tend to cancel each other out, driving the variance toward zero.
* **Zero Variance at Full Batch:** As seen in the plot at $N_{batch} = 200$ (the full dataset size $N$), the variance drops to effectively zero. This is because there is no longer any "random sampling"; every "sample" is identical to the full dataset, yielding the exact same deterministic gradient every time.

#### 2. Why SGD becomes more stable as $N_{batch}$ increases
* **Consistent Direction:** Stability in this context refers to how consistently the gradient vector points toward the true minimum of the loss function. With a higher $N_{batch}$, the calculated gradient $g_k$ is a more accurate approximation of the true gradient $\nabla_\Theta \mathcal{L}$.
* **Smoother Updates:** Because the variance is lower, the difference between consecutive gradient updates is smaller. This reduces the "jitter" or "zigzagging" effect seen in pure SGD, preventing the optimization algorithm from making erratic jumps in the wrong direction due to a single noisy outlier.

#### 3. The trade-off between stability and computational cost
* **Computational Cost:** Computing the gradient for a large batch requires performing matrix operations on large tensors. This increases the CPU/GPU memory requirement and the time required to compute a single update step.
* **Convergence Efficiency:** While small batches (low stability) are noisy, they are computationally cheap and allow for many updates per second. Large batches (high stability) provide accurate updates but are expensive.
* **The Sweet Spot (Mini-Batch):** The ideal trade-off is usually found in "Mini-Batch GD" (e.g., sizes 32 to 128). This range provides enough variance reduction to ensure stable convergence (as seen in the steep drop in your plot between sizes 1 and 20) while maintaining the computational speed of stochastic updates.

# Exercise 2.3

### 1. How noise helps escape shallow minima or bad regions

- **Breaking Symmetry and Saddle Points:** In the specific landscape of the function provided (which resembles a Rosenbrock function), there is a "ridge" or saddle point region near $\Theta\_1 = 0$. As seen in the trajectory with **No Noise (Blue line)**, standard Gradient Descent follows the exact gradient path. If initialized perfectly on the axis of symmetry (e.g., at $\Theta\_1 = 0$), the gradient with respect to $\Theta\_1$ is zero. The algorithm gets "stuck" on this ridge or moves extremely slowly because the deterministic gradient does not provide any lateral force to push it toward the basins of attraction at $\Theta\_1 = \pm 1$.
- **Exploration via Fluctuation:** The **Moderate Noise (Orange line)** and **High Noise (Green line)** introduce random perturbations ($\epsilon\_k$) to the update step. Even if the true gradient is zero (or very small), the random noise "kicks" the parameters sideways. Once the parameters are knocked off the unstable ridge, the gradient becomes non-zero and strong, quickly guiding the optimization into the deeper valleys towards the global minima (marked with red Xs). Essentially, noise turns the optimization into an exploration process that prevents the model from stalling in flat regions or saddle points.

### 2. How too much noise prevents convergence

- **Inability to Settle:** While noise is beneficial for exploration, it becomes detrimental during the exploitation phase (fine-tuning). As seen in the plot with **High Noise (Green line)**, particularly with the larger learning rate, the trajectory behaves erratically.
- **The "Bounce" Effect:** Near the minimum, the true gradient $\nabla \mathcal{L}$ approaches zero. However, the noise term $\epsilon\_k$ remains constant (with variance $\sigma^2=0.5$). Consequently, the update step is dominated by noise rather than the gradient signal. Instead of settling at the precise optimal point $[-1, 1]$, the parameters oscillate violently around it. This prevents the loss from stabilizing at the lowest possible value, as the algorithm constantly "overshoots" the target due to the high variance of the stochastic updates.

# Exercise 2.4

### 1. Why GD gives a smooth curve and SGD oscillates

The smoothness of the loss curve is determined by the consistency of the gradient calculation. In **Full Gradient Descent (Black line)**, the algorithm computes the gradient using the entire dataset ($N=1338$) for every single update. Because the data does not change between iterations, the calculated gradient vector points precisely in the direction of the steepest descent for the global loss surface. This results in a deterministic, monotonic decrease in error, creating the smooth curve visible in the plot.

In contrast, **SGD with Batch Size = 1 (Red line)** approximates the gradient using a single, randomly selected example. This individual sample may have a label ($y$) that is an outlier or simply noisy compared to the average. Consequently, the gradient calculated from it might point in a direction that increases the global error, even if it decreases the error for that specific point. This high variance in the gradient estimate causes the "zigzagging" or oscillation seen in the loss plot and the erratic spikes in the gradient norm.

### 2. Why larger batches reduce noise but cost more per iteration

There is a direct trade-off between statistical stability and computational effort. As observed in the results, **Batch Size = 50 (Blue line)** is significantly smoother and converges faster than **Batch Size = 1 (Red line)**. This happens because averaging gradients over 50 samples reduces the variance of the estimate (by a factor of roughly $1/\sqrt{50}$), effectively cancelling out the noise from individual outliers.

However, this stability comes at a cost. Computing the gradient for a batch of 50 requires 50 times more matrix multiplication operations (or vector dot products) than a batch of 1. While modern hardware (SIMD/GPUs) handles vectorization efficiently, making batches of 32-64 very fast, increasing the batch size indefinitely (up to Full GD) eventually leads to memory bottlenecks and diminishing returns in convergence speed per second of computation.

### 3. Why all methods roughly converge to the same region

The loss function for Linear Regression (Mean Squared Error) is **convex**, meaning it is shaped like a bowl with a single global minimum. Regardless of the path taken—whether it is the direct line of Full GD or the "drunk walk" of SGD—gravity (the gradient) eventually pulls the parameters toward the bottom of this bowl.

Comparing your final parameters confirms this:

- **Full GD:** `[0.001, 0.273, 0.168, 0.057]`
- **SGD (Batch=50):** `[-0.000, 0.279, 0.166, 0.056]`

These values are nearly identical because both methods reached the basin of the minimum. The slight deviation seen in **SGD Batch=1** (`[-0.107, ...]`) suggests that while it reached the correct *region*, the fixed learning rate was too high to allow it to settle perfectly into the minimum, causing it to bounce around the optimal point (noise floor).

### 4. Why SGD is more suitable for large datasets, even when noisy

While Full GD (Black line) is smooth, notice that it descended much slower than the SGD methods; even at epoch 200, it is just reaching the loss levels that SGD (Batch=50) reached around epoch 10.

For very large datasets (e.g., $N=1,000,000$), Full GD requires processing one million records just to take **one** step. In contrast, SGD updates the parameters after seeing only a few examples. If the dataset contains redundant information (common in large data), SGD can make thousands of useful updates and reach a "good enough" solution before Full GD has even finished its first epoch. The noise of SGD is a small price to pay for the massive speed advantage in terms of *updates per second*.