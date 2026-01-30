# Exercise 3.1
**. Discussion: Why is the decision boundary linear?**

- **Mathematical Justification:** Logistic Regression makes a prediction based on the probability $P(y=1|x) = \sigma(z)$, where $\sigma$ is the sigmoid function and $z = \Theta^T x$. The **decision boundary** is the threshold where the model is maximally uncertain, meaning the probability of being class 1 is exactly 0.5.

    Since $\sigma(0) = 0.5$, this boundary occurs precisely when the linear input $z$ is zero:

$$\Theta^T x = \theta\_0 + \theta\_1 x\_1 + \theta\_2 x\_2 = 0$$
- **Geometric Interpretation:** The equation above is a linear equation in the input variables $x\_1$ and $x\_2$. In 2D space, this defines a straight line. If we were in higher dimensions ($D&gt;2$), this equation would define a flat hyperplane. Therefore, despite using a non-linear activation function (sigmoid) to output probabilities, the boundary separating the classes remains linear in the feature space.

# Exercise 3.2

### 1. Theoretical Discussion

**Why do gradients become noisier for small batches?**

The gradient computed on a mini-batch is an **statistical estimate** of the true gradient (the gradient calculated on the entire dataset).

- **Mathematical Intuition:** If the dataset has variance $\sigma^2$, the variance of the mean of a batch of size $m$ is approximately $\frac{\sigma^2}{m}$.
- **Batch Size = 1 (SGD):** You are estimating the direction of steepest descent based on a single data point. If that point is an outlier or has a label that contradicts its neighbors (noise), the computed gradient will point in a wildly different direction than the true gradient. This high variance is what we call "noise."

**Why do larger batches give smoother curves?**

- **Averaging Effect:** As you increase the batch size (e.g., to 10 or to $N$), you are averaging the gradient over more samples. The random fluctuations ("noise") from individual data points tend to cancel each other out.
- **Convergence to Truth:** As the batch size approaches the total dataset size $N$ (Full GD), the estimate approaches the true gradient. Consequently, the update steps become consistent and deterministic, leading to a smooth, monotonic descent towards the minimum.

<head></head>
### 2. Comments on the Obtained Results

Your plots highlight a critical distinction between "convergence per epoch" and "stability."

- **Convergence Speed (Loss vs Epoch):**

    - **SGD (Batch=1) [Blue Line]:** It converges the fastest in terms of epochs. By the end of Epoch 0, it has already reached a near-optimal loss. This is because in one single epoch, SGD performs $N=400$ parameter updates (one for each point). Even though individual steps are noisy, the cumulative effect of 400 steps moves the model much further than the single step performed by Full GD.
    - **Full GD [Green Line]:** It is the slowest in terms of epochs. It makes only **one** update per epoch. It takes about 10-15 epochs to reach the accuracy that SGD reached in just 1 epoch.
- **Smoothness & Noise (Why does SGD look smooth here?):**

    - You might wonder why the SGD curve in your plot looks smooth and not "noisy" or "zig-zagging" as theory suggests.
    - **Reason:** Your code records the loss (`loss_hist.append`) only **at the end of each epoch**, after the inner loop has finished.

    ```python
    for _ in range(epochs):
    for i in range(0, N, batch_size):
        # ... update theta ...
    loss_hist.append(loss_f(theta, X, y)) # <--- Logged once per epoch
    ```

    If you were to plot the loss at every *step* (inside the inner loop), the Blue line (Batch=1) would look extremely jittery and noisy. However, looking at the aggregate result at the end of the epoch masks this noise, showing only the rapid convergence benefits of the frequent updates.
- **Accuracy:**

    All methods successfully reach 100% accuracy. SGD hits this target almost instantly due to the high frequency of updates, while Full GD takes a steadier, slower path.

# Exercise 3.3

### Discussion of Results

The results show near-perfect performance, which is expected given that the synthetic dataset generated in Exercise 1 consists of two well-separated Gaussian clusters. However, even with this "easy" dataset, we can observe the fundamental trade-offs in classification thresholds.

**1. How lower thresholds increase recall and lower precision**

- **Observation:** At a **threshold of 0.3**, your model is "liberal" or "optimistic" in predicting the positive class (Class 1). It classifies any sample with a probability $&gt;30\%$ as positive.
- **Result:** This strategy ensured **Recall was perfect (1.0)**, meaning no positive cases were missed (FN=0). However, it allowed one "False Positive" to slip through (FP=1), which slightly lowered the **Precision to 0.995**.
- **Theory:** Lowering the threshold reduces the standard of evidence required to predict "Positive." This catches more true positives (increasing Recall) but inevitably captures more noise or negative instances (increasing False Positives and lowering Precision).

**2. How higher thresholds increase precision and reduce recall**

- **Observation:** When you raised the **threshold to 0.7**, the model became "conservative" or "strict." It only predicted "Positive" if it was extremely confident ($&gt;70\%$ probability).
- **Result:** This strictness filtered out the single False Positive encountered earlier (FP dropped to 0). Consequently, **Precision increased to a perfect 1.0**. While Recall usually drops with higher thresholds, in this specific dataset the positive examples were so distinct that they all remained correctly classified (Recall stayed at 1.0).
- **Theory:** Raising the threshold acts as a filter that removes uncertain predictions. This minimizes False Positives (maximizing Precision) but increases the risk of rejecting actual positive cases that are subtle or weak (increasing False Negatives and lowering Recall).

**3. Why classification metrics depend on the application**

As discussed in class, there is no single "best" threshold; it depends entirely on the cost of errors in the specific real-world domain:

- **High Precision Preference (High Threshold):** In **Spam Detection**, we want to avoid False Positives at all costs (e.g., sending an important work email to the spam folder). We prefer to let some spam through (lower Recall) rather than lose legitimate mail.
- **High Recall Preference (Low Threshold):** In **Medical Diagnosis** (e.g., screening for a tumor), a False Negative is dangerous (telling a sick patient they are healthy). We lower the threshold to catch every possible case (High Recall), accepting that this will lead to more False Positives (healthy people sent for further testing).

# Exercose 3.4

## Discussion of the results

<head></head>
Here is the consolidated answer incorporating all the requested considerations regarding the experiment on the Pima Indians Diabetes dataset.

### 1. Why Normalization is Required

In the data preprocessing step, standardizing the features using `(X - mean) / std` is critical for three main reasons:

- **Conditioning of the Loss Surface:** Real-world features have vastly different ranges (e.g., "Age" $\approx 20-80$ vs. "Insulin" $\approx 0-800$). Without standardization, the loss function contours form highly elongated ellipses (deep, narrow valleys). Gradient Descent struggles in this landscape, bouncing back and forth across the valley walls instead of moving down the valley floor. Standardization makes the contours more spherical (well-conditioned), allowing the gradient to point more directly toward the minimum.
- **Stable Optimization:** With unscaled data, weights associated with large inputs (like Insulin) would need to be very small, while weights for small inputs (like Age) would need to be large. A single global learning rate $\eta$ cannot satisfy both requirements simultaneously, leading to instability or divergence.
- **Meaningful Gradient Magnitudes:** Standardization ensures that all features contribute approximately equally to the initial gradient updates, preventing one feature from dominating the learning process simply because it has larger numerical values.

### 2. SGD vs. Adam Comparison

Looking at the comparison plots and the final metrics, we observe distinct behaviors:

- **Convergence Speed:**

    **Adam (Orange line)** converges significantly faster than **SGD (Blue line)**. In the Loss plot, Adam drops vertically in the first 10-20 epochs, reaching a loss of $\approx 0.50$ almost immediately. SGD, conversely, follows a slow, linear descent and barely reaches a loss of $\approx 0.51$ after 200 epochs.
- **Oscillation:**

    While SGD (Batch=32) appears relatively stable due to batch averaging, it is "slow" to navigate the curvature. Adam appears smoother in its descent because it effectively dampens oscillations in high-variance directions (via the momentum term $m\_t$) and accelerates in flat directions (via the variance term $v\_t$).
- **Adaptive Learning Rates:**

    The superior performance of Adam illustrates the power of adaptive learning rates.

    - **SGD** uses a fixed learning rate $\eta$ for all parameters. If $\eta$ is small enough to be stable for the most sensitive parameter, it is often too small for the others, leading to slow convergence.
    - **Adam** computes an individual effective learning rate for each parameter $\theta\_j$ by dividing by $\sqrt{\hat{v}\_t}$. This allows it to take large steps for parameters with small gradients (flat regions) and small, cautious steps for parameters with large gradients (steep regions), navigating the complex landscape of the Diabetes dataset much more efficiently.

### 3. Final Model Evaluation

The quantitative results confirm Adam's superiority on this task:

- **Accuracy:** Adam achieved **78.26%**, beating SGD's **76.43%**.
- **F1 Score:** Adam achieved **0.6514** vs. SGD's **0.6373**.
- **Precision:** Adam was notably more precise (**0.7393** vs. **0.6883**), meaning it produced fewer False Positives (55 vs. 72). This is often desirable in medical diagnostics to avoid unnecessary alarm, even though the Recall was slightly lower.

**Conclusion:** For real-world datasets with heterogeneous features, adaptive optimizers like Adam generally offer a better trade-off between speed and stability compared to vanilla SGD, minimizing the need for extensive hyperparameter tuning.

## 5th point answer

<head></head>
#### 1. Which method converges faster?

**Adam converges significantly faster.**

Looking at the "Loss: SGD vs Adam" plot, the difference is stark.

- **Adam (Orange Line):** The loss drops vertically, reaching a value below 0.55 within the first 20 epochs and stabilizing around 0.47 shortly after.
- **SGD (Blue Line):** The loss decreases linearly and very slowly. Even after 200 epochs, it has only reached $\approx 0.51$, a value that Adam surpassed in its first few iterations.

    In terms of accuracy, Adam jumps to $\approx 77\%$ almost immediately, whereas SGD takes the full 200 epochs to slowly climb to $\approx 76\%$.

#### 2. Which oscillates more?

- **SGD (in this specific case):** Appears very smooth, but this is deceptive. The smoothness here is due to the learning rate ($10^{-3}$) being too small for the flat regions of the loss surface. SGD is taking tiny, consistent steps down a long slope, never moving fast enough to "oscillate" across valley walls.
- **Adam:** Shows some jitter (oscillation) in the accuracy plot once it reaches the plateau. This is expected behavior for a fast optimizer: it quickly reaches the minimum and then "bounces" slightly around the optimal parameters because it is sensitive to the noise in the mini-batches. However, it does not suffer from the detrimental "zigzag" oscillation of simple SGD because momentum ($m\_t$) smoothes the trajectory.

#### 3. Relation to Adaptive Learning Rates (Class Theory)

The superior performance of Adam relates directly to the two adaptive components discussed in class:

- **Momentum ($\hat{m}\_t$):**

    Adam maintains a moving average of past gradients (momentum). This helps the model "gain speed" in directions where the gradient consistently points the same way, pushing it through flat plateaus where standard SGD would stall. This explains why Adam drops so fast in the early epochs compared to SGD.
- **RMSProp / Adaptive Scaling ($\frac{1}{\sqrt{\hat{v}\_t} + \epsilon}$):**

    This is the crucial "adaptive learning rate" component.

    - Standard SGD uses the same scalar learning rate $\eta$ for all parameters. If one feature has a steep gradient and another has a flat gradient, a single $\eta$ cannot satisfy both (it will be too slow for one or unstable for the other).
    - Adam divides the update by the square root of the accumulated squared gradients ($\sqrt{\hat{v}\_t}$).

        - For parameters with small gradients (flat regions): $\hat{v}\_t$ is small $\rightarrow$ the step size is **boosted** (division by a small number).
        - For parameters with large gradients (steep regions): $\hat{v}\_t$ is large $\rightarrow$ the step size is **dampened** (division by a large number).

**Conclusion:** Adam creates a custom learning rate for each parameter. It was able to identify that the "Diabetes" dataset loss landscape had varying curvatures and automatically boosted the step size for the slow parameters, allowing it to converge in a fraction of the time it took SGD.