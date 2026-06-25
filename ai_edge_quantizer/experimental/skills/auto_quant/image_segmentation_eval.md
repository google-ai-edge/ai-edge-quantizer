# Image Segmentation Evaluation in AEQ

When utilizing the AI Edge Quantizer (AEQ) for models executing image
segmentation, agents must employ distinct validation protocols. Standard
classification evaluation metrics like Top-1 Accuracy or Cross-Entropy fail to
capture the severity of spatial degradation inherent to segmentation masking.

## 1. Segmentation Error Projection
Classification models reduce multi-dimensional feature graphs into standard
logits; noise typically lowers confidence slightly without shattering the
underlying categorical prediction. Segmentation networks process dense spatial
tensors resulting in strict 2D layouts.
Numerical quantization degradation in a segmentation model does not just lower a
prediction—it destroys the physical output, resulting in structural "ghosting",
shredded boundary edges, and background-foreground inversion.
Agents MUST evaluate using strict End-to-End visual metrics beyond base tensor
SNR and MSE.

## 2. The Golden Rule: SNR for Search, Output MSE for Stopping, Masks for Illustration
Because segmentation networks can be massive, generating a post-processed
end-to-end spatial mask to evaluate every single layer ablation is
computationally prohibitive and unnecessary for the search logic.

* **Internal Tensor Search (Fast)**: Rely on standard internal tensor SNR
(`ValidationErrorMetric.SNR`) in AEQ's `validate()` to rapidly rank and identify
the most mathematically sensitive internal layers.

* **Stopping Criteria (Efficient)**: Use the **Output Tensor MSE** (the MSE of
the model's raw final output tensor, retrievable directly via AEQ's
`ValidationErrorMetric.MSE`) to determine if your loop iterations satisfy the
overall degradation bounds.
* **End-to-End Visual Metrics (For Illustration Only)**: True Task Metrics on
the post-processed masks (like Mask MSE, mIoU, Dice Coefficient, or Heatmaps)
should only be computed at the very end of an experiment for reporting and
illustration purposes. Do not use post-processed Mask MSE as the mathematical
stopping condition in your search loop.

### End-to-End Visual Metrics (Illustration)
When presenting the final model quality against the Float32 ground truth in
reports, use:

* **Mask Mean Squared Error (MSE)**: Safest continuous metric for assessing
global intensity degradation.
* **Mean Intersection over Union (mIoU)**: Evaluates strict edge boundary
retention.
* **Dice Coefficient (F1 Score)**: Essential if the model masks highly
imbalanced image foregrounds.

### Example Boilerplate: Mask MSE metric
In AEQ, you can define custom metric functions evaluated during validation.
Calculate the MSE difference against the original model directly:
```python
import numpy as np

def calculate_mask_mse(float_interpreter, quant_interpreter, input_data):
    # Set input tensors
    float_interpreter.set_tensor(input_details[0]['index'], input_data)
    float_interpreter.invoke()
    float_mask = float_interpreter.get_tensor(output_details[0]['index'])
    
    quant_interpreter.set_tensor(quant_input_details[0]['index'], input_data)
    quant_interpreter.invoke()
    quant_mask = quant_interpreter.get_tensor(quant_output_details[0]['index'])
    
    # Calculate Spatial difference
    return np.mean(np.square(float_mask - quant_mask))
```

## 3. Visualization & Figure Generation
Agents must render visual differences into exported graphs rather than just
logging a decimal number.
**Critical Setup**: You must use `plt.savefig()` and avoid `plt.show()` when
validating in headless mode.

### Example Plotting Pipeline (Difference Heatmap)
```python
import matplotlib.pyplot as plt
import numpy as np

def save_fidelity_heatmap(test_image, float_mask, quantized_mask, save_path):
    diff_mask = np.abs(float_mask - quantized_mask)[0, :, :, 0] # Expand dims
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Base Truth
    axes[0].imshow(float_mask[0, :, :, 0], cmap='gray')
    axes[0].set_title('FP32 Baseline Mask')
    axes[0].axis('off')
    
    # Quantized Mask
    axes[1].imshow(quantized_mask[0, :, :, 0], cmap='gray')
    axes[1].set_title('Quantized Output')
    axes[1].axis('off')

    # Difference Heatmap
    heatmap = axes[2].imshow(diff_mask, cmap='hot')
    axes[2].set_title('Absolute Difference')
    axes[2].axis('off')
    plt.colorbar(heatmap, ax=axes[2], fraction=0.046, pad=0.04)

    # Save visualization for Output Report
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
```
