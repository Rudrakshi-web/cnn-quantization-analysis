# CNN Quantization Analysis

This project presents a comparative study of Direct and Gradual Quantization techniques applied to a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset. The goal is to evaluate the impact of reduced numerical precision on model performance and generalization.

---

## Overview

Deep neural networks often require significant computational and memory resources. Quantization is a model compression technique that reduces the precision of weights and activations, enabling more efficient inference.

This project investigates two quantization strategies:

- Direct Quantization: All model parameters are quantized at once without retraining  
- Gradual Quantization: Model parameters are quantized progressively with intermediate retraining  

---

## Objectives

- Develop a CNN model for CIFAR-10 image classification  
- Apply low-bit quantization (2-bit, 3-bit, and 4-bit)  
- Compare the effects of direct and gradual quantization  
- Analyze performance using standard evaluation metrics  

---

## Model Architecture

The CNN architecture consists of:

- Three convolutional layers with 32, 64, and 128 filters  
- ReLU activation functions  
- Max pooling layers for spatial downsampling  
- Fully connected layers (256 neurons followed by 10 output classes)  
- Dropout (0.5) to reduce overfitting  

---

## Methodology

### Direct Quantization
All model weights are quantized in a single step. This approach is computationally efficient but often leads to significant accuracy degradation due to abrupt information loss.

### Gradual Quantization
Quantization is applied incrementally, layer by layer. After each quantization step, the model is retrained to adapt to the reduced precision. This approach helps preserve performance and improves stability.

---

## Results

### Bit-width vs Accuracy
![Bit vs Accuracy](results/graphs/bit_vs_accuracy.png)

### Training Curve
![Training Curve](results/graphs/training_curve.png)

### Confusion Matrix
![Confusion Matrix](results/graphs/confusion_matrix.png)

---

## Performance Summary

| Method   | 2-bit | 3-bit | 4-bit |
|----------|------|------|------|
| Direct   | ~10% | ~55% | ~65% |
| Gradual  | ~74% | ~79% | ~79% |

---

## Key Findings

- Gradual quantization significantly outperforms direct quantization at all tested bit-widths  
- It preserves model accuracy even under aggressive compression  
- In some cases, it improves generalization compared to the full precision model  
- The process behaves similarly to regularization by introducing controlled perturbations  

---

## Key Insight

Gradual quantization enables the model to adapt to precision constraints incrementally, reducing the negative impact of quantization noise and preventing performance collapse.

---

## Dataset

CIFAR-10 dataset
Automatically downloaded using torchvision
Not included in the repository to maintain a lightweight project

---

## Evaluation Metrics

Accuracy
Confusion Matrix
Precision, Recall, and F1-score

---

## Limitations

The model architecture is relatively simple and not state-of-the-art
Experiments are limited to the CIFAR-10 dataset
No deployment or hardware-level benchmarking is included

---

## Future Work

Implement quantization-aware training (QAT)
Evaluate performance on deeper architectures such as ResNet
Explore deployment on edge devices

---

## Tech Stack

Python
PyTorch
NumPy
Matplotlib
Scikit-learn

---

## Conclusion

This project demonstrates that gradual quantization is a more effective approach for compressing neural networks compared to direct quantization. It maintains high accuracy at low bit-widths and provides improved generalization through incremental adaptation.

---