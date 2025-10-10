<div align="center">

# Model Card for InceptionResnetV1 (FaceNet) — Face Recognition on CASIA-WebFace  

[![Model](https://img.shields.io/badge/Model-InceptionResnetV1-blue.svg)](https://github.com/timesler/facenet-pytorch)  
[![Dataset](https://img.shields.io/badge/Dataset-CASIA--WebFace-orange.svg)](https://www.kaggle.com/datasets/debarghamitraroy/casia-webface)  

</div>

---

## Model Details

**Model:** InceptionResnetV1 (FaceNet, facenet-pytorch)  
**Task:** Face Recognition / Verification  
**Training:** Fine-tuned on CASIA-WebFace (pretrained on VGGFace2)  
**Framework:** PyTorch 2.x  
**License:** Apache 2.0 (FaceNet) + dataset-dependent  

InceptionResnetV1 is a deep CNN designed for face embedding extraction.  
It is widely used for face verification and identification tasks, leveraging triplet loss to learn discriminative facial representations.

---

## Uses

### Direct Use
- Benchmarking on face recognition tasks  
- Feature extraction and embedding generation  
- Face verification and identification in research settings  

### Downstream Use
- Fine-tuning for verification (e.g., contrastive or triplet loss)  
- Transfer learning to datasets like LFW or VGGFace2  

### Out-of-Scope Use
- Biometric surveillance or identification without consent  

---

## Training Details

- **Dataset:** CASIA-WebFace (pipeline splits: train/val/test, stratified by identity)  
- **Preprocessing:** Resize (160×160), normalize to [-1, 1]  
- **Optimizer:** Adam (lr=0.0001)  
- **Loss:** TripletMarginLoss (margin=0.2)  
- **Batch Size:** 64  
- **Epochs:** 20  
- **Hardware:** Apple MPS or CPU  

---

## Evaluation

- **Validation Split:** 15% (pipeline-generated)  
- **Metrics:** Top-1 nearest neighbor accuracy  
- **Expected Accuracy:** To be determined (see eval logs)  

---

## Technical Specifications

| Parameter      | Value                |
|----------------|---------------------|
| Architecture   | InceptionResnetV1   |
| Params         | ~24M                |
| Input Size     | 160×160             |
| Embedding Size | 512                 |
| Inference      | <50 ms (GPU), <500 ms (CPU) |

---

## Citation

```bibtex
@article{schroff2015facenet,
  title={FaceNet: A Unified Embedding for Face Recognition and Clustering},
  author={Schroff, Florian and Kalenichenko, Dmitry and Philbin, James},
  journal={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={815--823},
  year={2015}
}
```

## Maintainers

- Team XOXO — UPC MLOps Project 2025
- Contact: nashly.erielis.gonzalez@estudiantat.upc.edu

---