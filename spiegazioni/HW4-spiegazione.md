# Homework 4

## Exercise 4.1


### 1. Code & Reconstruction Verification

Your implementation correctly performs the SVD decomposition and verification.

- **Reconstruction Error:** You obtained a Frobenius norm error of approximately `7.71e-15`.
- **Conclusion:** This value is effectively zero within the limits of machine precision (floating-point arithmetic). This numerically proves that $A$ can be perfectly reconstructed as the product $U \Sigma V^T$, satisfying the definition of SVD

### 2. Discussion of Singular Values

**Why do singular values appear in descending order?**

- **Definition:** By mathematical definition and convention in algorithms (like `np.linalg.svd`), singular values are always ordered such that $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$<source-footnote ng-version="0.0.0-PLACEHOLDER" _nghost-ng-c1831576998=""></source-footnote>.<sources-carousel-inline ng-version="0.0.0-PLACEHOLDER" _nghost-ng-c2676262027=""><!----><source-inline-chips _ngcontent-ng-c2676262027="" _nghost-ng-c3554374441="" class="ng-star-inserted"><source-inline-chip _ngcontent-ng-c3554374441="" _nghost-ng-c2970764686="" class="ng-star-inserted"></source-inline-chip></source-inline-chips></sources-carousel-inline>


- **Maximizing Variance:** The first singular value $\sigma_1$ captures the direction of maximum variance (most information) in the data. The second, $\sigma_2$, captures the maximum remaining variance orthogonal to the first, and so on. This ordering is crucial for tasks like compression, as it allows us to truncate the series at $k$ and know we have kept the "best" possible $k$-dimensional approximation.

**Why do small singular values correspond to "less important" directions?**

- **Sum of Dyads:** The matrix $A$ can be written as a weighted sum of rank-1 matrices (dyads):

$$A = \sum_{i=1}^{r} \sigma_i u_i v_i^T$$
Here, $\sigma_i$ acts as the **weight** or importance score for the $i$-th component

- **Information Content:** If $\sigma_i$ is large, the term $u_i v_i^T$ contributes significantly to the structure of $A$. If $\sigma_i$ is very small, that term adds very little detail (often just noise). Removing terms with small singular values results in a compressed matrix $A_k$ that is still very close to the original $A$ in terms of distance (Frobenius norm)
**Why does floating-point arithmetic make exact zeros rare?**

- **Numerical Noise:** In theory, if a matrix has rank  $r < \min(m, n)$, the singular values $\sigma_{r+1}$ onwards should be exactly $0$.
- **Reality:** In Python (and all computers), numbers are represented with finite precision (IEEE 754 standard). The tiny errors accumulated during the random generation of numbers and the iterative calculation of SVD mean that a theoretical "zero" often appears as a very small number (e.g., $10^{-16}$) rather than a hard $0$. This is why we often look for a "gap" in singular values to determine the **numerical rank** rather than looking for exact zeros.

## Exercise 4.2

**Explain why SVD gives the *optimal* rank-k approximation.**

The SVD provides the optimal rank-$k$ approximation because of the **Eckart-Young-Mirsky Theorem**. Here is the intuition based on the provided text:

1. **Decomposition into Information Content:** The SVD allows us to write the matrix $A$ as a sum of rank-1 matrices (dyads):

$$A = \sum_{i=1}^{r} \sigma_i u_i v_i^T$$

where the scalar $\sigma_i$ represents the "weight" or **importance** of the $i$-th component <source-footnote ng-version="0.0.0-PLACEHOLDER" _nghost-ng-c1783869716=""></source-footnote>.<sources-carousel-inline ng-version="0.0.0-PLACEHOLDER" _nghost-ng-c1881558635=""><!----><source-inline-chips _ngcontent-ng-c1881558635="" _nghost-ng-c3554374441="" class="ng-star-inserted"><source-inline-chip _ngcontent-ng-c3554374441="" _nghost-ng-c2970764686="" class="ng-star-inserted"></source-inline-chip></source-inline-chips></sources-carousel-inline>


1. **Ordered Importance:** Since singular values are always sorted in descending order ($\sigma_1 \ge \sigma_2 \ge \dots \ge 0$), the first term $\sigma_1 u_1 v_1^T$ captures the most "energy" (variance) of the matrix, the second captures the next most, and so on.
2. **Minimizing the Residual:** The error introduced by truncating the series at $k$ (i.e., approximating $A$ with $A_k$) is determined entirely by the singular values we *discarded*. Specifically, the Frobenius error is the square root of the sum of the squared discarded singular values:

$$||A - A_k||_F = \sqrt{\sum_{i=k+1}^{r} \sigma_i^2}$$

To make this error as small as possible, we must discard the *smallest* possible $\sigma$ values. SVD guarantees this by keeping the largest $\sigma$ values (indices $1$ to $k$) and discarding the tail (indices $k+1$ to $r$). This ensures that $A_k$ is mathematically the closest rank-$k$ matrix to $A$.

## Exercise 4.3


### Discussion of Results

**1. How visual quality improves with k**

- **Low k (e.g., k=5):** The image is extremely blurry and blocky. Only the grossest features (light sky vs. dark coat) are visible. This represents the "mean" structure of the image without details.
- **Medium k (e.g., k=20 to 50):** Structural elements like the tripod legs and the camera shape become recognizable. The "ghosting" artifacts reduce significantly.
- **High k (e.g., k=100 to 200):** The image becomes nearly indistinguishable from the original to the human eye. Fine details, such as the texture of the grass, are restored.

**2. Why most of the "energy" is contained in the first singular values**

- **Information Concentration:** Singular values $\sigma_i$ represent the "importance" or information content of the corresponding rank-1 layer (dyad)
- **Redundancy in Data:** Real-world images (like the Cameraman) are highly structured; adjacent pixels are likely to be similar. This correlation means that a few dominant "patterns" (the first few singular vectors) can describe the majority of the image variance. The remaining thousands of singular values mostly capture high-frequency noise or very subtle details

**3. The trade-off between compression and fidelity**

- There is a direct inverse relationship: as **rank k** increases, the **fidelity** (visual quality) improves (error drops), but the **compression ratio** worsens (file size grows).
- **The Sweet Spot:** The goal is to find a $k$ where the error is low enough that the image looks good, but $k$ is still small enough to save significant space. In your results, **k=50** offers a strong balance: the image is clearly recognizable, yet you still save **80.45%** of the storage space.

**4. The connection between SVD and optimal low-rank approximation**

- The Eckart-Young-Mirsky theorem states that the matrix $X_k$ constructed by truncating the SVD to the top $k$ singular values is the **optimal** rank-$k$ matrix in terms of minimizing the error 


- Mathematically, no other matrix of rank $k$ can produce a smaller Frobenius norm difference $||X - X_k||_F$ than the one produced by SVD. This guarantee makes SVD the "gold standard" for linear dimensionality reduction and compression tasks.

## Exercise 4.5

<head></head>
### 1. Analysis of Digits 3 vs. 4 (2D Projection)

- **Separation:** In the first scatter plot, we observe two distinct clusters. The digit **3** (yellow/lighter points) and the digit **4** (purple/darker points) are separated reasonably well along the first Principal Component (PC1).
- **Overlap:** There is a noticeable overlap region in the center where the clusters merge. This indicates that while the global structure of a "3" differs from a "4", there are handwriting variations (e.g., sloppy writing) that make them pixel-wise similar in the lowest dimensions.
- **Generalization:** The test set points (marked with 'x' in) map consistently onto the regions defined by the training set. This confirms that the principal components learned from the training data successfully capture the true, invariant features of the digits, rather than just memorizing the training samples.

### 2. Comparison with Other Digit Pairs

The effectiveness of PCA depends heavily on the geometric distinctness of the classes:

- **Digits 1 vs. 7:**

    These digits separate relatively well, forming curved, elongated clusters. This is expected as "1" and "7" are visually distinct (straight line vs. angled corner), though slanted writing can cause some overlap.
- **Digits 5 vs. 8:**

    These clusters show **significant overlap** and are much harder to separate. "5" and "8" share many geometric features (loops and curves in similar locations), making them difficult to distinguish using only linear projections like PCA. This suggests that non-linear manifold learning (like t-SNE) or higher dimensions ($k&gt;2$) would be required for clear separation.

### 3. Analysis of 3D Projection ($k=3$)

- **Observation:** The 3D scatter plot adds the third principal component (PC3) to the visualization.
- **Impact:** Adding dimensions helps resolve ambiguities. Points that appear to overlap in the 2D plane might be separated along the vertical Z-axis (PC3). This illustrates the trade-off in dimensionality reduction: reducing $d$ too much (e.g., to 2) sacrifices information that might be crucial for separating similar classes, while keeping more components ($k=3$ or higher) retains more variance and separability.

## Exercise 4.6

<head></head>
### **1. Code & Results Evaluation**

**Step 3: Centroid Classifier**

- **Vectorization:** You used `np.linalg.norm(..., axis=1)` correctly. This is much faster than looping through rows.
- **Broadcasting:** The line `Z_test - mu_3` effectively subtracts the centroid from every test sample simultaneously. This is the "Pythonic" way to handle this math.
- **Results:**

    - **Linear Accuracy:** 97.39%
    - **Centroid Accuracy:** 97.33%
    - **Observation:** The results are nearly identical. This implies that for digits 3 and 4, the clusters in PCA space are roughly **spherical** and have **similar spreads (variance)**. When two clusters are perfect spheres of equal size, the optimal decision boundary *is* the perpendicular bisector of the line connecting their centroids. Your Linear Classifier found essentially this same line.

**Step 4: Comparison Visualization**

- **The "Error" Plot:** This is the highlight of your work. By plotting the "Green Circles" (misclassified points) on top of the boundary and centroids, you instantly show *where* the model fails:

    - The failures are exclusively in the "twilight zone" between the two clusters.
    - There are no outliers deep inside the wrong cluster, which suggests the data is clean.

### **2. Addressing the "Repeat with other digits" section**

The exercise asks you to repeat this for `{(1,7), (5,8), (2,3)}` and observe the changes. Since you haven't run those yet, here is a guide on **what to look for and how to interpret it** when you do.

The relationship between **Cluster Shape** and **Classifier Performance** will change depending on the digits:

#### **Case A: Digits 1 vs 7 (Structural Similarity)**

- **Expectation:** These might be harder to separate than 3 vs 4.
- **Cluster Shape:** The digit "1" often varies mostly in *angle* (slant). This creates a long, thin, elliptical cluster in PCA space (like a cigar).
- **Centroid vs. Linear:**

    - If the "1" cluster is very elongated, the **Centroid Classifier** might fail. It assumes clusters are round. It might misclassify a "1" that is far from the center (a very slanted "1") as a "7".
    - The **Linear Classifier** usually adapts better here because it cares about the *boundary*, not the center.

#### **Case B: Digits 5 vs 8 (Topological Similarity)**

- **Expectation:** This is often the hardest pair in MNIST because 5s and 8s share similar bottom loops and curvature.
- **Cluster Separation:** You will likely see the red and blue clouds **overlapping** significantly in the center.
- **Decision Boundary:** The boundary will cut through a dense region of points.
- **Metrics:** Accuracy will likely drop (perhaps to ~85-90% instead of 97%). You will see many more "Green Circles" (errors) mixed into the clusters.

#### **Case C: Digits 2 vs 3**

- **Expectation:** Moderately distinct.
- **Observation:** Look at the **Centroid Separation**. If the centroids are far apart relative to the spread of the data, accuracy will remain high.

### **3. Summary of Key Concepts**

When you run those experiments, you are effectively testing **Geometry vs. Learning**:

| **Feature** | **Centroid Classifier** | **Linear Classifier (Logistic Regression)** |
| --- | --- | --- |
| **Model Type** | **Generative / Geometric** | **Discriminative** |
| **Assumption** | Assumes classes are **spherical blobs** with equal variance. | Assumes classes are separated by a **flat plane** (line). |
| **Training** | Fast (just compute mean). No optimization loops. | Slower (requires Gradient Descent loops). |
| **Best for...** | Well-separated, round clusters. | Elongated clusters or touching clusters where the boundary isn't in the middle. |
