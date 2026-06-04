# LightQANet

This is the official PyTorch codes for the paper: [LightQANet: Quantized and Adaptive Feature  Learning for Low-Light Image Enhancement](https://ieeexplore.ieee.org/abstract/document/11417255)

>This paper has been accepted to the IEEE Transactions on Multimedia (TMM) 2026.

<img src="images/framework5.pdf" width="800px">

## Abstract:
Low-light image enhancement (LLIE) aims to improve illumination while preserving high-quality color and texture. However, existing methods often fail to extract reliable feature representations due to severely degraded pixel-level information under low-light conditions, resulting in poor texture restoration, color inconsistency, and artifact.
To address these challenges, we propose LightQANet, a novel framework that introduces quantized and adaptive feature learning for low-light enhancement, aiming to achieve consistent and robust image quality across diverse lighting conditions.
From the static modeling perspective, we design a Light Quantization Module (LQM) to explicitly extract and quantify illumination-related factors from image features. By enforcing structured light factor learning, LQM enhances the extraction of light-invariant representations and mitigates feature inconsistency across varying illumination levels.
From the dynamic adaptation perspective, we introduce a Light-Aware Prompt Module (LAPM), which encodes illumination priors into learnable prompts to dynamically guide the feature learning process. LAPM enables the model to flexibly adapt to complex and continuously changing lighting conditions, further improving image enhancement.
Extensive experiments on multiple low-light datasets demonstrate that our method achieves state-of-the-art performance, delivering superior qualitative and quantitative results across various challenging lighting scenarios.

## Experiments:
### LSRW
<img src="images/LSRW.png" width="800px">

### Real World
<img src="images/real_world.png" width="800px">

## Dependencies and Installation

- CUDA >= 11.0
- Other required packages in `codeenhance.yaml`

