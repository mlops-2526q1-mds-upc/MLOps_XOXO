<div align="center">    

# Model Card for Hierarchical Geolocation Estimation with Scene Classification    

[![Places365](https://img.shields.io/badge/Model-Places365-blue.svg)](https://github.com/CSAILVision/places365)  
[![Paper Places365](https://img.shields.io/badge/Paper-Places365-green.svg)](http://places2.csail.mit.edu/PAMI_places.pdf)  
[![GitHub](https://img.shields.io/badge/Code-GeoEstimation-black.svg)](https://github.com/TIBHannover/GeoEstimation/tree/master?tab=readme-ov-file)
[![Conference](http://img.shields.io/badge/ECCV-2018-4b44ce.svg)](https://openaccess.thecvf.com/content_ECCV_2018/papers/Eric_Muller-Budack_Geolocation_Estimation_of_ECCV_2018_paper.pdf)   

</div>

---

## Table of Contents
- [Model Details](#model-details)
  - [Model Description](#model-description)
  - [Model Sources](#model-sources)
- [Uses](#uses)
  - [Direct Use](#direct-use)
  - [Downstream Use](#downstream-use)
  - [Out-of-Scope Use](#out-of-scope-use)
- [Bias, Risks, and Limitations](#bias-risks-and-limitations)
- [How to Get Started with the Model](#how-to-get-started-with-the-model)
- [Training Details](#training-details)
  - [Training Data](#training-data)
  - [Training Procedure](#training-procedure)
  - [Preprocessing](#preprocessing)
  - [Training Hyperparameters](#training-hyperparameters)
  - [Speeds, Sizes, Times](#speeds-sizes-times)
- [Evaluation](#evaluation)
  - [Testing Data](#testing-data)
  - [Factors](#factors)
  - [Metrics](#metrics)
  - [Results](#results)
- [Model Examination](#model-examination)
- [Environmental Impact](#environmental-impact)
- [Technical Specifications](#technical-specifications)
  - [Model Architecture and Objective](#model-architecture-and-objective)
  - [Compute Infrastructure](#compute-infrastructure)
  - [Hardware](#hardware)
  - [Software](#software)
- [Citation](#citation)
- [Glossary](#glossary)
- [More Information](#more-information)
- [Model Card Authors](#model-card-authors)
- [Model Card Contact](#model-card-contact)

---

## Model Details

### Model Description
This model is a **hierarchical geolocation estimation system** that predicts the approximate location of a photo based solely on its pixels. It treats geolocation as a **classification problem**, where the Earth is subdivided into geographical cells and the model predicts the most likely cell for a given image.  

The model integrates two key innovations:  
1. **Hierarchical multi-partitioning:** Multiple spatial resolutions are used simultaneously, from coarse (continent-level) to fine (city-level). This allows the model to leverage both global and local geographic cues.  
2. **Scene classification integration:** The model uses scene information (indoor, natural, urban) predicted with a ResNet pretrained on **Places365**, which helps disambiguate difficult cases and improves accuracy.  

- **Developed by:** Müller-Budack, Pustu-Iren, and Ewerth (TIB Hannover & Leibniz University Hannover). Adapted and extended by **Team XOXO (UPC MLOps Project 2025)**.  
- **Funded by:** German Research Foundation (DFG), project EW 134/4-1.  
- **Shared by:** TIB Hannover research group.  
- **Model type:** Deep convolutional neural network (classification).  
- **Language(s):** Not applicable (vision-only, no text).  
- **License:** Academic research use (inherits licenses from Im2GPS/YFCC100M dataset and Places365 model).  
- **Finetuned from model:** ResNet (pretrained on ImageNet, extended with Places365 scene labels).  

### Model Sources
- **Repository (GeoEstimation):** [GeoEstimation GitHub](https://github.com/TIBHannover/GeoEstimation)  
- **Repository (Places365):** [Places365 GitHub](https://github.com/CSAILVision/places365)  
- **Paper (GeoEstimation):** [ECCV 2018: Geolocation Estimation of Photos](https://openaccess.thecvf.com/content_ECCV_2018/papers/Eric_Muller-Budack_Geolocation_Estimation_of_ECCV_2018_paper.pdf)  
- **Paper (Places365):** [Places: A 10 Million Image Database for Scene Recognition](http://places2.csail.mit.edu/PAMI_places.pdf)  
- **Demo:** [GeoEstimation Demo](https://tibhannover.github.io/GeoEstimation/)  

---

## Uses

### Direct Use
- Benchmarking on photo geolocation tasks.  
- Research in computer vision, geolocation, and multimedia retrieval.  
- Applications that need approximate photo location predictions at multiple scales.  

### Downstream Use
- Fine-tuning on specific geographic regions or domains.  
- Transfer to related tasks such as **landmark recognition** or **scene understanding**.  

### Out-of-Scope Use
- Personal tracking or surveillance of individuals.  
- Security-sensitive or military applications.  
- Real-time navigation systems requiring exact coordinates.  

---

## Bias, Risks, and Limitations
- **Bias:** The dataset (YFCC100M subset / Im2GPS) contains many more photos from urban, tourist-heavy areas than from rural or underrepresented regions.  
- **Risks:** Photos may contain identifiable individuals or private spaces; predictions may be misinterpreted as precise.  
- **Limitations:** Accuracy drops for ambiguous natural scenes (forests, deserts). Predictions are probabilistic classifications, not exact GPS coordinates.  

### Recommendations
- Use strictly for academic or research purposes.  
- Communicate uncertainty clearly when presenting predictions.  

---

## How to Get Started with the Model

Example (PyTorch workflow):

```python
import torch
import torchvision.models as models
from PIL import Image
from torchvision import transforms

# Load pretrained ResNet
model = models.resnet152(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, num_geo_cells)

# Preprocess image
img = Image.open("sample.jpg")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
input_tensor = transform(img).unsqueeze(0)

# Inference
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1)
```

## Training Details

### Training Data

The training data is a subset of the **Yahoo Flickr Creative Commons 100M (YFCC100M)** dataset, specifically the version introduced for the **MediaEval Placing Task 2016 (MP-16)**. This subset contains approximately 4.7 million geo-tagged Flickr images. Test data included the **Im2GPS benchmark (237 images)** and **Im2GPS3k benchmark (2,997 images)**.

### Training Procedure

#### Preprocessing

- Images resized to 224×224 pixels.  
- Geo-cells generated with the **S2 geometry library**, balancing samples per cell.  
- Scene labels assigned from **Places365 categories**.  

#### Training Hyperparameters

- Optimizer: SGD with momentum (0.9).  
- Learning rate: 0.01 with exponential decay (factor 0.5 every 5 epochs).  
- Weight decay: 0.0001.  
- Batch size: 128 (used consistently in training).  
- Epochs: ~15 + fine-tuning for 5.  
- Loss: Cross-entropy (geo-cell + scene labels for MTN).  

#### Speeds, Sizes, Times

- Training took several days on a multi-GPU cluster (NVIDIA Tesla).  
- Input size: 224×224 px.  
- Number of geo-cells varied with partitioning (3,298 coarse, 7,202 middle, 12,893 fine).  

---

## Evaluation

### Testing Data

- **Im2GPS benchmark:** 237 images.  
- **Im2GPS3k benchmark:** 2,997 images.  

### Factors

- Evaluation disaggregated by scene type (indoor, natural, urban).  
- Performance compared across coarse, middle, fine, and hierarchical partitions.  

### Metrics

- **Great-Circle Distance (GCD)** in km.  
- Accuracy reported at thresholds: 1 km, 25 km, 200 km, 750 km, 2,500 km.  

### Results

- The hierarchical approach outperformed state-of-the-art methods (PlaNet, Im2GPS).  
- Accuracy improved particularly for **urban and indoor scenes** with Individual Scene Networks (ISNs).  

#### Summary

The hierarchical + scene-based approach consistently improved results on both Im2GPS and Im2GPS3k, demonstrating better generalization with fewer training images compared to PlaNet.  

---

## Model Examination

The model’s predictions depend strongly on visual distinctiveness. Scene classification helps reduce ambiguity, but interpretability is limited because decisions are encoded in deep CNN features.  

---

## Environmental Impact

- **Hardware Type:** Multi-GPU cluster (NVIDIA Tesla).  
- **Hours used:** Several days of training (~15 epochs + fine-tuning).  
- **Cloud Provider:** Not specified; academic cluster.  
- **Compute Region:** Germany.  
- **Carbon Emitted:** Not reported, but comparable to other CNN training on millions of images.  

---

## Technical Specifications

### Model Architecture and Objective

- **Base:** ResNet-101/152.  
- **Extensions:** Additional fully connected layers for geo-cell classification; auxiliary head for scene classification.  
- **Objective:** Multi-class classification (geo-cell prediction + optional scene classification).  

### Compute Infrastructure

- **Frameworks:** TensorFlow (original), PyTorch (re-implementations).  

#### Hardware

- Multi-GPU training with batch size of 128. 

#### Software

- Python 3, TensorFlow, PyTorch, torchvision, S2 Geometry library.  

---

## Citation

**BibTeX:**
```bibtex
@inproceedings{muller2018geolocation,
  title={Geolocation Estimation of Photos using a Hierarchical Model and Scene Classification},
  author={Müller-Budack, Eric and Pustu-Iren, Kader and Ewerth, Ralph},
  booktitle={European Conference on Computer Vision (ECCV)},
  pages={563--579},
  year={2018}
}
```

**APA:**  
Müller-Budack, E., Pustu-Iren, K., & Ewerth, R. (2018). Geolocation estimation of photos using a hierarchical model and scene classification. *European Conference on Computer Vision (ECCV)*, 563–579.  

---

## Glossary

- **Geo-cell:** A subdivision of the Earth into discrete regions used as classification labels.  
- **Scene classification:** Assigning labels like indoor, natural, or urban to images.  
- **Great-Circle Distance (GCD):** The shortest distance between two points on Earth’s surface.  
- **ISN (Individual Scene Network):** A CNN trained specifically for a scene type.  
- **MTN (Multi-Task Network):** A CNN jointly trained for geolocation and scene classification.  

---

## More Information

- Places365 project: [CSAIL MIT Places365](https://github.com/CSAILVision/places365)  
- GeoEstimation project: [TIB Hannover GeoEstimation](https://github.com/TIBHannover/GeoEstimation)   

---

## Model Card Authors

Prepared by **Team XOXO (UPC MLOps Project 2025):**

- Mateja Zatezalo — GitHub: `matzatezalo` — Email: mateja.zatezalo@estudiantat.upc.edu  
- Pawarit Jamjod — GitHub: `meenpawarit` — Email: pawarit.jamjod@estudiantat.upc.edu  
- Albert Puiggros Figueras — GitHub: `apuiggros` — Email: albert.puiggros@estudiantat.upc.edu  
- Nashly González — GitHub: `NashGG47` — Email: nashly.erielis.gonzalez@estudiantat.upc.edu  

---

## Model Card Contact

- [GeoEstimation GitHub Repository](https://github.com/TIBHannover/GeoEstimation)  
- [ECCV 2018 Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Eric_Muller-Budack_Geolocation_Estimation_of_ECCV_2018_paper.pdf)  
- [Im2GPS Dataset (CMU)](http://graphics.cs.cmu.edu/projects/im2gps/)  

