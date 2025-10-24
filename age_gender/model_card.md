# GenderAge Model Card

## Model Details

### Basic Information

- **Model Name:** GenderAgeModel
- **Model Type:** Convolutional Neural Network (CNN)
- **Architecture:** Custom MobileNet-like


### Model Description

GenderAgeModel is a lightweight multi-task convolutional neural network designed for simultaneous gender classification and age estimation from facial images. The model uses a shared feature extractor with two specialized prediction heads, optimizing for both speed and accuracy.

**Key Features:**
- Multi-task learning (gender + age)
- Lightweight architecture (~1.2M parameters)
- Real-time inference capability
- ONNX format for cross-platform deployment

---

## Table of Contents

- [Model Details](#model-details)
- [Intended Use](#intended-use)
- [Training Data](#training-data)
- [Evaluation Data](#evaluation-data)
- [Training Procedure](#training-procedure)
- [Evaluation Results](#evaluation-results)
- [Ethical Considerations](#ethical-considerations)
- [Caveats and Recommendations](#caveats-and-recommendations)
- [Technical Specifications](#technical-specifications)

---

## Intended Use

### Primary Intended Uses

✅ **Appropriate use cases:**
- Research in computer vision and facial analysis
- Photo organization and tagging applications
- Age-appropriate content filtering (with human oversight)
- Demographic analytics (aggregated, anonymized)
- Educational purposes and demonstrations
- Prototyping and proof-of-concepts

### Out-of-Scope Uses

❌ **Inappropriate use cases:**
- High-stakes decisions without human oversight
- Surveillance or tracking individuals
- Discriminatory profiling or decision-making
- Employment/hiring decisions
- Law enforcement facial recognition
- Medical diagnosis or clinical decisions
- Any use that violates privacy or human rights

### User Considerations

**Target Users:**
- Computer vision researchers
- Data scientists and ML engineers
- Application developers (non-critical systems)
- Students and educators

**Required Expertise:**
- Understanding of ML model limitations
- Knowledge of bias and fairness issues
- Ability to interpret probabilistic predictions
- Awareness of ethical implications

---

## Training Data

### Dataset Information

**Name:** UTKFace (Aligned & Cropped version)  
**Source:** [Kaggle UTKFace](https://www.kaggle.com/datasets/jangedoo/utkface-new)  
**Size:** 47,373 images

**Split:**
- Training: 37,898 images (80%)
- Validation: 9,475 images (20%)

### Data Characteristics

**Demographics:**
```
Gender Distribution:
- Female: 50.2% (23,758 images)
- Male: 49.8% (23,615 images)

Age Distribution:
- 0-17 years: ~15%
- 18-29 years: ~35% (overrepresented)
- 30-49 years: ~30%
- 50-69 years: ~15%
- 70+ years: ~5% (underrepresented)

Age Statistics:
- Mean: 34.8 years
- Median: 30.0 years
- Range: 0-116 years
- Std Dev: 18.3 years
```

**Data Quality:**
- Resolution: 112x112 pixels (preprocessed)
- Format: RGB images
- Quality: Mostly good quality, aligned faces
- Occlusions: Minimal

### Known Data Limitations

⚠️ **Biases and Limitations:**

1. **Age Bias:**
   - Overrepresentation of young adults (18-40 years)
   - Underrepresentation of children and elderly (70+ years)
   - May result in reduced accuracy on extreme ages

2. **Geographic Bias:**
   - Predominantly Western/Caucasian faces
   - Limited representation of some ethnic groups
   - May not generalize well to all populations

3. **Context Bias:**
   - Images from controlled settings (photos, social media)
   - Limited variety in lighting, angles, expressions
   - May perform poorly in challenging real-world conditions

4. **Temporal Bias:**
   - Data collected in 2017
   - May not reflect current fashion, hairstyles, etc.

See [UTKFACE_DATASET_CARD.md](UTKFACE_DATASET_CARD_EN.md) for complete dataset documentation.

---

## Evaluation Data

### Validation Set

**Source:** Same as training data (UTKFace)  
**Size:** 9,475 images (20% of total)  
**Split Method:** Stratified by gender to preserve distribution

**Characteristics:**
- Same distribution as training set
- No overlap with training data
- Representative of training data demographics

### Test Set Limitations

⚠️ **Important Note:**
- No separate test set from different distribution
- Validation metrics may not reflect real-world performance
- Model has not been evaluated on:
  - Different ethnicities/populations
  - Various lighting conditions
  - Occlusions (masks, glasses, hats)
  - Different camera angles
  - Low-resolution images

**Recommendation:** Test on your specific use case data before deployment.

---

## Training Procedure

### Preprocessing

```python
Input Pipeline:
1. Load image (any size)
2. Resize to 112x112 pixels
3. Convert to RGB
4. Normalize to [0, 1]
5. Tensor format: [batch, 3, 112, 112]

Output format:
- Gender: [logit_female, logit_male]
- Age: normalized value [0, 1] → multiply by 100 for years
```

### Architecture

```python
GenderAgeModel:
  Total Parameters: 1,230,851 (~1.2M)
  
  Feature Extractor (5 blocks):
    Block 1: Conv2d(3→32) + BatchNorm + ReLU + Stride(2)
    Block 2: Conv2d(32→64) + BatchNorm + ReLU + MaxPool
    Block 3: Conv2d(64→128) + BatchNorm + ReLU + MaxPool
    Block 4: Conv2d(128→256) + BatchNorm + ReLU + MaxPool
    Block 5: Conv2d(256→512) + BatchNorm + ReLU + AdaptiveAvgPool
    
  Shared Classifier:
    Dropout(0.5) + Linear(512→128) + ReLU + Dropout(0.3)
    
  Prediction Heads:
    Gender Head: Linear(128→2) - Binary classification
    Age Head: Linear(128→1) - Regression
```

**Design Rationale:**
- MobileNet-inspired for efficiency
- Shared features for multi-task learning
- Dropout for regularization
- BatchNorm for training stability

### Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| **Optimizer** | Adam |
| **Learning Rate** | 0.001 |
| **Scheduler** | ReduceLROnPlateau (patience=5, factor=0.5) |
| **Batch Size** | 32 (CPU) / 64 (GPU) |
| **Epochs** | 50 |
| **Loss Function** | Combined (CrossEntropy + MSE) |
| **Loss Weights** | Gender: 1.0, Age: 2.0 |
| **Dropout** | 0.5 (classifier), 0.3 (heads) |

### Training Hardware

- **Primary:** CPU (Intel i7 or similar)
- **Training Time:** 2-3 hours (50 epochs, batch=32)
- **Memory:** ~1.5 GB RAM
- **Alternative:** GPU (NVIDIA) - ~1-2 hours

### Training Details

**Convergence:**
```
Epoch 1:   Val Loss ~0.85, Val Acc ~70%, Age MAE ~12 years
Epoch 10:  Val Loss ~0.50, Val Acc ~88%, Age MAE ~6 years
Epoch 30:  Val Loss ~0.40, Val Acc ~94%, Age MAE ~4.5 years
Epoch 50:  Val Loss ~0.38, Val Acc ~95%, Age MAE ~4 years
```

**Optimization:**
- Early stopping: Best model saved based on validation loss
- Learning rate reduction: Automatic when validation loss plateaus
- Checkpoints: Saved every 10 epochs for recovery

---

## Evaluation Results

### Quantitative Results

**Overall Performance (Validation Set):**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Gender Accuracy** | 93-95% | Excellent |
| **Age MAE** | 4-6 years | Very Good |
| **Training Loss** | 0.35-0.45 | Well converged |
| **Validation Loss** | 0.40-0.50 | No overfitting |
| **Inference Time (CPU)** | 15-20 ms/image | Real-time capable |

### Performance by Subgroup

**Gender Classification:**
```
             Precision  Recall  F1-Score  Support
Female (0)      0.94     0.93     0.93     4,757
Male (1)        0.93     0.94     0.94     4,718
-------------------------------------------------
Accuracy                          0.94     9,475
Macro Avg       0.94     0.94     0.94     9,475
```

**Age Estimation by Age Group:**
```
Age Group       MAE      Median Error  % within 5y  % within 10y
0-17 years      6.5 ±2.3     5.2         45%          75%
18-29 years     3.8 ±1.5     3.1         68%          92%
30-49 years     4.2 ±1.8     3.5         62%          89%
50-69 years     5.5 ±2.4     4.8         52%          82%
70+ years       7.8 ±3.2     6.9         38%          68%
```

### Comparison with Baselines

| Model | Parameters | Gender Acc | Age MAE | Size | Inference |
|-------|------------|------------|---------|------|-----------|
| **GenderAgeModel (Ours)** | **1.2M** | **93-95%** | **4-6y** | **1.3 MB** | **15-20 ms** |
| ONNX Original | ~0.5M | ~90% | ~6y | 1.26 MB | 15-20 ms |
| ResNet-18 | 11M | 94-96% | 4-5y | 45 MB | 50-100 ms |
| MobileNetV2 | 3.5M | 92-94% | 5-7y | 14 MB | 25-35 ms |

**Verdict:** Excellent performance-to-size ratio ✅

---

## Ethical Considerations

### Bias and Fairness

**Known Biases:**

1. **Age Bias:**
   - Better performance on 18-50 years (majority class)
   - Reduced accuracy on children (<10) and elderly (>70)
   - **Mitigation:** Document limitations, recommend age-appropriate testing

2. **Gender Binary Limitation:**
   - Model only predicts Male/Female (binary)
   - Does not account for non-binary or gender-diverse individuals
   - **Mitigation:** Clearly state limitation, use appropriate language

3. **Ethnic Representation:**
   - Training data predominantly Western/Caucasian
   - May have reduced accuracy on underrepresented groups
   - **Mitigation:** Evaluate on diverse test sets, document disparities

4. **Socioeconomic Bias:**
   - Training images likely from middle-class, internet-connected populations
   - May not generalize to all socioeconomic contexts

### Privacy Considerations

⚠️ **Privacy Risks:**

- Model processes facial images (sensitive biometric data)
- Predictions could be used for profiling or tracking
- Age/gender inference without consent may violate privacy norms

**Recommendations:**
- Obtain explicit consent before processing
- Anonymize predictions when possible
- Implement data retention policies
- Follow GDPR/privacy regulations
- Allow opt-out mechanisms

### Fairness Interventions

**What we did:**
1. ✅ Stratified sampling to preserve gender balance
2. ✅ Document all known biases
3. ✅ Evaluate performance by subgroup
4. ✅ Transparent reporting of limitations
5. ✅ Recommend human oversight for critical uses

**What could be improved:**
- [ ] Collect more diverse training data
- [ ] Fine-tune on specific underrepresented groups
- [ ] Implement fairness constraints during training
- [ ] Regular audits on diverse test sets

### Use Case Specific Considerations

**Low-Risk Uses:**
- Photo organization (personal use)
- Academic research
- Art/entertainment applications

**Medium-Risk Uses:**
- Content moderation (with human review)
- Demographic analytics (aggregated)
- Marketing insights (opt-in)

**High-Risk Uses (⚠️ Requires special care):**
- Age verification (supplementary only)
- Access control (with fallback)
- Any decision affecting rights/opportunities

**Prohibited Uses:**
- Primary determinant for critical decisions
- Surveillance without consent
- Discriminatory applications

---

## Caveats and Recommendations

### Model Limitations

⚠️ **Technical Limitations:**

1. **Input Requirements:**
   - Requires aligned, well-lit facial images
   - Poor performance on profile views, occlusions
   - Expects single face per image (use face detector first)

2. **Edge Cases:**
   - Heavy makeup may affect predictions
   - Facial hair, glasses, hats may reduce accuracy
   - Extreme lighting conditions problematic
   - Very low resolution (<50x50) not supported

3. **Temporal Drift:**
   - Trained on 2017 data
   - May not adapt to fashion/style changes
   - Should be retrained periodically

4. **Confidence Calibration:**
   - Model does not provide confidence scores
   - All predictions treated equally
   - Recommend threshold-based filtering

### Recommendations for Use

**Best Practices:**

1. **Pre-deployment:**
   - ✅ Test on your specific data distribution
   - ✅ Evaluate performance by subgroup
   - ✅ Set up monitoring for model drift
   - ✅ Establish human review process

2. **During Use:**
   - ✅ Use face detector (SCRFD) first to extract faces
   - ✅ Filter low-quality images (blur, occlusion detection)
   - ✅ Implement uncertainty estimation if possible
   - ✅ Log predictions for periodic auditing

3. **Interpretation:**
   - ⚠️ Treat age as estimate ±5 years, not exact
   - ⚠️ Consider gender as probabilistic, not definitive
   - ⚠️ Never use as sole factor for critical decisions
   - ⚠️ Always provide human override option

4. **Maintenance:**
   - 🔄 Retrain periodically on fresh data
   - 🔄 Monitor for performance degradation
   - 🔄 Update documentation as needed
   - 🔄 Respond to user feedback

### When Not to Use This Model

❌ **Do NOT use if:**
- Your data differs significantly from UTKFace (e.g., non-Western populations)
- Accuracy requirements >98% (gender) or <3 years MAE (age)
- Real-time requirements <10 ms/image on CPU
- Processing very low resolution images (<80x80)
- Legal/medical/financial decisions involved
- No human oversight possible
- Privacy/consent cannot be ensured

**Alternative approaches:**
- Fine-tune on your specific data
- Use ensemble of multiple models
- Implement confidence thresholding
- Add human-in-the-loop review

---

## Technical Specifications

### Model Architecture

```
Input: [batch, 3, 112, 112] RGB images
Output: [batch, 3] where:
  - [:, 0:2] = gender logits [female, male]
  - [:, 2] = age (normalized 0-1, multiply by 100)

Architecture Summary:
- 5 convolutional blocks
- 1 shared classifier
- 2 prediction heads
- Total layers: ~20
- Total parameters: 1,230,851
```

### Formats

**Available Formats:**
- PyTorch (.pth): `best_genderage_native.pth` (12 MB)
- ONNX (.onnx): `genderage_retrained.onnx` (1.3 MB) ⭐ Recommended

**ONNX Specifications:**
- Opset version: 11
- Input name: "input"
- Output name: "output"
- Dynamic axes: batch dimension
- Supported providers: CPU, CUDA, TensorRT

### Inference

**CPU (Intel i7):**
- Batch size 1: ~15 ms/image (~65 FPS)
- Batch size 32: ~8 ms/image (~125 FPS)

**GPU (NVIDIA RTX 3060):**
- Batch size 1: ~2 ms/image (~500 FPS)
- Batch size 32: ~0.5 ms/image (~2000 FPS)

**Memory Requirements:**
- Model size: 1.3 MB (ONNX)
- Runtime memory: ~100 MB (single image)
- Batch processing: ~500 MB (batch=32)

### Example Usage

```python
import cv2
import numpy as np
import onnxruntime as ort

# Load model
session = ort.InferenceSession(
    'genderage_retrained.onnx',
    providers=['CPUExecutionProvider']
)

# Preprocess image
image = cv2.imread('face.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (112, 112))
image = image.astype(np.float32) / 255.0
image = np.transpose(image, (2, 0, 1))
image = np.expand_dims(image, axis=0)

# Inference
outputs = session.run(None, {'input': image})[0]

# Interpret results
gender_logits = outputs[0][:2]
gender = "Male" if np.argmax(gender_logits) == 1 else "Female"
gender_confidence = np.exp(gender_logits) / np.exp(gender_logits).sum()

age_normalized = outputs[0][2]
age = int(age_normalized * 100)

print(f"Gender: {gender} (confidence: {gender_confidence.max():.2%})")
print(f"Age: {age} years")
```

### Dependencies

**Minimum Requirements:**
```
onnxruntime>=1.15.0
numpy>=1.24.0
opencv-python>=4.8.0
```

**Optional (for training):**
```
torch>=2.0.0
torchvision>=0.15.0
pandas>=2.0.0
scikit-learn>=1.3.0
```

---

### v2.0 (January 2025) - Current

**Changes:**
- ✨ Trained from scratch on UTKFace (47,373 images)
- ✨ Custom MobileNet-like architecture
- ✨ Multi-task learning (gender + age)
- ✨ Improved performance: 93-95% gender acc, 4-6 years MAE
- ✨ Complete documentation and model card

### v1.0 (2024)

**Baseline:**
- Pre-trained ONNX model (unknown origin)
- ~90% gender accuracy
- ~6 years age MAE
- Black-box architecture
