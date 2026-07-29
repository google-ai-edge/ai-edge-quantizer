# Pareto Curve Visualization Best Practices

When rendering the final Pareto frontier graph in an AEQ Exploration report,
agents must consistently format their `matplotlib` code to produce readable,
analytically useful charts. Given the massive scale differences between baseline
models and failed empirical combinations, naive plotting quickly results in
illegible graphs.

Please adhere to the following best practices when generating the
`pareto_curve.png` artifact:

## 1. Metric Selection & Axes
The user prioritizes the functional End-to-End quality over internal tensor
fidelity.

* **Primary Graph**: The X-axis must always be **Model Size (MB)**. The Y-axis
must always be the **Primary Task Metric**.
* **Dual Metrics**: If you choose to plot an internal
tensor metric (like SNR) alongside the functional metric, you MUST use dual
Y-axes (`ax.twinx()`). You must explicitly differentiate them using distinct
markers and colors and provide a clear legend. Be warned: dual axes graphs can
easily become visually cluttered.

## 2. Managing Scale and Outliers
During your empirical search loop, you will inevitably generate "broken"
configurations (e.g., a blind INT8 config producing massive noise like 10,000+
MSE). Plotting these linearly will compress the legitimate, cluster-optimized
models on the frontier into a single overlapping dot at the bottom of the graph.

* **Logarithmic Scaling**: You MUST use a logarithmic scale for your Y-axis if
the validation error spans multiple orders of magnitude. Because the FP32
baseline will often have `0.0` error, use `symlog` (Symmetrical Log) to safely
plot zero without crashing, or manually clip `0` to a small epsilon.
    ```python
    ax.set_yscale('symlog', linthresh=0.01)
    ```

## 3. Drawing the Actual Frontier Line
A collection of scatter points is not technically a Pareto curve until the
non-dominated frontier line is explicitly drawn!
You must mathematically identify the non-dominated points (configs where no
other config exists that is both smaller *and* has higher accuracy), sort them
by size, and draw a distinct line connecting them.

```python
# Identifying the Pareto Frontier (Assuming lower error is better)
# `data` is a list of tuples: (size_mb, error, name)
data.sort(key=lambda x: x[0]) # Sort by Size ascending
pareto_points = []
best_error = float('inf')

for size, error, name in data:
    if error < best_error:
        pareto_points.append((size, error))
        best_error = error

# Draw the step line connecting the optimal threshold
pareto_sizes, pareto_errors = zip(*pareto_points)
ax.plot(pareto_sizes, pareto_errors, color='red', linestyle='--', label='Pareto Frontier')
```

## 4. Preventing Annotation Clutter
If you generate 10+ intermediate loop versions (e.g., `ver1` through `ver5`),
they will likely cluster at nearly identical coordinates, causing `plt.annotate`
text labels to instantly over-write each other into a black, illegible blob.

* **Annotation Selection**: **DO NOT** annotate every single dot. Only annotate
the Baseline, the extreme limits, and your explicitly recommended Top 3 profiles
(e.g., "Target", "Balanced", "Size").
* **Offsetting**: Visually offset the text slightly using `xytext=(5, 5)`
combined with `textcoords='offset points'`.

