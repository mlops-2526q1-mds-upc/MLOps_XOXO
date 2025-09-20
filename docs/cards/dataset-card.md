---
pretty_name: "Im2GPS"
dataset_type: "image"
task_categories:
  - other:image-geolocation
task_ids:
  - image-geolocation
size_categories:
  - 1M<n<10M
license: "Creative Commons (user-defined, Flickr)"
homepage: "http://graphics.cs.cmu.edu/projects/im2gps/"
repository: "https://graphics.cs.cmu.edu/projects/im2gps/flickr_code.html"
paper: "https://graphics.cs.cmu.edu/projects/im2gps/im2gps.pdf"
leaderboard: "http://graphics.cs.cmu.edu/projects/im2gps/results.html"
language:
  - en
tags:
  - computer-vision
  - geolocation
  - image-classification
  - benchmark
  - flickr
  - gps
  - exif
configs:
  - name: im2gps-train
    description: "Training split of ~6.47 million Flickr photos with GPS metadata"
    split: train
    size: 6472304
  - name: im2gps-test
    description: "Evaluation split of 237 manually filtered geotagged images"
    split: test
    size: 237
version: "1.0.0"
release_date: "2008-06-01"
dataset_creators:
  - name: "James Hays"
    affiliation: "Carnegie Mellon University (at the time), now Brown University"
  - name: "Alexei A. Efros"
    affiliation: "Carnegie Mellon University (at the time), now UC Berkeley"
maintainers:
  - name: "Team XOXO - UPC MLOps Project 2025"
    email: "nashly.erielis.gonzalez@estudiantat.upc.edu"
    github: "https://github.com/NashGG47"
citation:
  bibtex: |
    @inproceedings{hays2008im2gps,
      title={IM2GPS: estimating geographic information from a single image},
      author={Hays, James and Efros, Alexei A},
      booktitle={2008 IEEE Conference on Computer Vision and Pattern Recognition},
      pages={1--8},
      year={2008},
      organization={IEEE}
    }
dataset_summary: |
  Im2GPS is a large-scale dataset of 6.5 million geo-tagged Flickr photos collected
  to investigate whether geographic location can be estimated directly from an image’s
  pixels. Each image is labeled with latitude and longitude coordinates derived from
  Flickr’s GPS metadata. The dataset was introduced in the CVPR 2008 paper by James Hays
  and Alexei A. Efros. It provides a training set of approximately 6.47 million photos
  and a benchmark test set of 237 manually curated images. Evaluation is performed using
  the great-circle distance (GCD) metric at thresholds of 200 km, 750 km, and 2500 km,
  corresponding to city/region-, country-, and continent-level accuracy.
---


<div align="center">    

# Dataset Card for Im2GPS — A Benchmark for Image Geolocation Estimation     

[![Dataset](https://img.shields.io/badge/Dataset-Im2GPS-brightgreen.svg)](http://graphics.cs.cmu.edu/projects/im2gps/)  
[![Paper](https://img.shields.io/badge/Paper-CVPR%202008-blue.svg)](https://graphics.cs.cmu.edu/projects/im2gps/im2gps.pdf)  

</div>

---

## Table of Contents
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
    - [Initial Data Collection and Normalization](#initial-data-collection-and-normalization)
    - [Who are the source data producers?](#who-are-the-source-data-producers)
  - [Annotations](#annotations)
    - [Annotation Process](#annotation-process)
    - [Who are the annotators?](#who-are-the-annotators)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

---

## Dataset Description

- **Homepage:** The official dataset homepage is the CMU Im2GPS project page, which describes the dataset, its motivation, and provides access to related materials: http://graphics.cs.cmu.edu/projects/im2gps/

- **Repository:** The original download scripts and related code for collecting Flickr images (“Flickr code”) are provided by the authors here: https://graphics.cs.cmu.edu/projects/im2gps/flickr_code.html

- **Paper:** The dataset and its methodology were introduced in the CVPR 2008 paper “IM2GPS: estimating geographic information from a single image,” available at: https://graphics.cs.cmu.edu/projects/im2gps/im2gps.pdf

- **Leaderboard:** Historical benchmark results and evaluation figures for Im2GPS are documented on the project’s results page: https://graphics.cs.cmu.edu/projects/im2gps/results.html

- **Point of Contact:** contact information as listed in the original CVPR 2008 paper: James Hays (jhhays@cs.cmu.edu), Alexei Efros (efros@eecs.berkeley.edu) 

### Dataset Summary
The Im2GPS dataset is a large-scale collection of **6.5 million geo-tagged Flickr photos**, each labeled with GPS coordinates. The dataset was created to explore the challenging computer vision problem of estimating the geographic location of a photo based solely on its pixel content, without relying on metadata such as captions or tags.  

This dataset provides the foundation for research in image-based geolocation, where the task is to predict where a photo was taken by analyzing its visual features. It also enables researchers to measure geolocation accuracy at different scales (continent, country, region, city) and to investigate how image content relates to secondary geographic attributes such as land cover type, population density, and urban versus rural classification.  

### Supported Tasks and Leaderboards
The dataset supports several main tasks:

- **Image Geolocation Estimation (task-category: other:image-geolocation)**:  
  This dataset can be used to train models that predict the geographic location of a photo based only on its visual content. The task consists of providing GPS coordinates for an input image. Success on this task is typically measured using the **Great-Circle Distance (GCD)** between predicted and ground-truth coordinates. In the original CVPR 2008 evaluation, performance is summarized at distance thresholds of **200 km**, **750 km**, and **2500 km**, reflecting city/region-, country-, and continent-level accuracy, respectively.  
  The best-known baseline models include nearest-neighbor approaches and later CNN-based methods (such as PlaNet and GeoEstimation). Historical benchmark results for these methods are available on the [Im2GPS results page](http://graphics.cs.cmu.edu/projects/im2gps/results.html).  
  Although the dataset does not maintain an active online leaderboard today, the official results page provides reference comparisons that serve the same purpose.  

- **Secondary Geographic Tasks (task-category: other:geographic-attributes)**:  
  Beyond direct location prediction, the dataset also supports secondary tasks derived from estimated locations. These include predicting **land cover classes** (such as forest, water, savanna), **population density**, **elevation gradients**, and distinguishing between **urban and rural areas**. These tasks are evaluated by comparing estimated geographic attributes (derived from predicted coordinates) against ground-truth geographic maps.  

---

### Languages

- **Language(s):** English (`en`) for the dataset documentation and project pages.  
  The dataset instances themselves are **images without textual annotations**; therefore, no natural language content is involved in the labels.    

---

## Dataset Structure

### Data Instances

Each data instance in the Im2GPS dataset consists of a photo and its associated geographic coordinates (latitude and longitude). The dataset itself is not distributed as a single archive of images. Instead, the authors provide **download scripts** and **text files** containing Flickr photo IDs with their GPS metadata. Researchers must use the provided code to download the original JPEG images from Flickr.  

For clarity, the following JSON-style block is an **illustrative example** of how a single instance is structured, even though the dataset itself is stored as JPEG files plus metadata lists:

```json
{
  "image": "photo_000123.jpg",
  "latitude": 48.8566,
  "longitude": 2.3522
}
```

In this example, the image file `photo_000123.jpg` is a JPEG taken in Paris, France. The latitude and longitude are floating-point numbers representing the location in decimal degrees.

### Additional Notes on Data Relationships

- Each image is independent and does not have explicit links to other data points in the dataset. There are no sequential or temporal relationships between images.  
- The only meaningful relationship between data points is **geographic proximity**. For instance, multiple images taken in the same region will have similar latitude and longitude values.  
- The test set of 237 images was curated to ensure that none of the test images come from the same photographers as the training images. This prevents overlap between splits and ensures fair evaluation.  
- Metadata beyond GPS coordinates (such as image titles, Flickr tags, or camera EXIF data) was excluded from the dataset in order to focus solely on the geolocation task using visual content.  

---

### Data Fields

The dataset provides the following fields:

- **`image`**: The actual photograph, stored in JPEG format. This field is the **input** to geolocation models.  
- **`latitude`**: A floating-point number indicating the latitude coordinate of where the photo was taken, expressed in decimal degrees. This field is part of the **output label** for geolocation tasks.  
- **`longitude`**: A floating-point number indicating the longitude coordinate of where the photo was taken, expressed in decimal degrees. This field is also part of the **output label** for geolocation tasks.  

No other textual or categorical annotations are included. The dataset is designed to emphasize purely **visual cues** from images.  

---

### Data Splits

The dataset is divided into a large training database and a smaller test benchmark:  

- **Training database:** Approximately **6,472,304 Flickr photos**, each resized to a maximum of 1024 pixels on the longer side and compressed in JPEG format. The total storage size is about 1 terabyte.  
- **Test set:** An evaluation benchmark of **237 carefully selected photos**. These images were originally sampled as **400 random geotagged Flickr images**, then **manually filtered** to remove low-quality, abstract, heavily edited, black-and-white, or privacy-sensitive photos. After filtering, 237 images remained.  

To ensure independence between splits, photos from the same Flickr users who contributed to the training database were **excluded** from the test set.  

Evaluation on the test set is performed by comparing predicted GPS coordinates against ground-truth values using the **Great-Circle Distance (GCD)** metric. Results are summarized at thresholds of **200 km**, **750 km**, and **2500 km**, corresponding to city/region-, country-, and continent-level accuracy, respectively.  

| Split    | Number of Images | Description                                                                 |
|----------|------------------|-----------------------------------------------------------------------------|
| Training | ~6.47 million    | Large-scale set of Flickr photos with GPS coordinates, resized and compressed |
| Test     | 237              | Randomly sampled and manually curated benchmark for evaluation              |

---

## Dataset Creation

### Curation Rationale

The Im2GPS dataset was created to address a fundamental research question in computer vision: *can the geographic location of a photo be estimated using only its pixels, without relying on metadata such as captions or tags?* The authors wanted to build a dataset that was global in scope, covering a diverse set of environments, and large enough to allow the development and benchmarking of geolocation models. The choice to use Flickr was motivated by its vast collection of user-contributed photos with GPS information, which provided sufficient scale to study this problem worldwide.

### Source Data

#### Initial Data Collection and Normalization

The dataset was constructed by downloading photos from Flickr that contained both GPS metadata and geographic keywords. Keywords included the names of countries, cities, and landmarks. Ambiguous or overly broad search terms (such as “Asia” or “Canada”) were excluded, and non-geographic terms (such as “birthday,” “concert,” or “abstract”) were filtered out.  

All collected photos were resized to a maximum resolution of 1024 pixels and compressed in JPEG format. The final dataset contains approximately **6.5 million photos**, totaling around 1 terabyte of storage. A separate test set of 237 images was sampled and manually filtered to exclude low-quality or privacy-sensitive images.  

#### Who are the source data producers?

The images were produced by individual Flickr users worldwide. Each photo was taken by a human photographer and uploaded voluntarily to the Flickr platform. The dataset creators did not control who the photographers were, and no demographic or personal information about the photo owners is provided. The conditions under which the images were created are therefore highly varied and reflect the diverse practices of Flickr’s user community. Some images may depict individuals or private spaces, but this information is incidental to the dataset and not the focus of the task.

### Annotations

#### Annotation process

The dataset does not include manual annotations created by the authors or crowdworkers. Instead, labels are automatically derived from the GPS metadata embedded in the images by the cameras or devices used. The latitude and longitude values associated with each photo serve as the ground-truth labels for the geolocation task. No additional annotation guidelines, validation steps, or inter-annotator agreement measures were necessary, since the data is machine-generated.

#### Who are the annotators?

There were no human annotators for this dataset. The only annotations consist of GPS coordinates that were automatically generated by the devices used by Flickr users and retained in the photo’s EXIF metadata. The dataset creators simply filtered and organized this information to produce the final dataset.

### Personal and Sensitive Information

The dataset may include photos of people, private properties, or sensitive locations, since Flickr users upload a wide range of content. GPS coordinates are sensitive because they reveal exact geographic positions. To reduce risks, the authors curated the test set to remove obviously problematic images, such as close-up photos of individuals. However, because the training set is very large, it is likely that some images still include sensitive content. The dataset should therefore be used with caution, and always under the assumption that it is for academic research only.

---

## Considerations for Using the Data

### Social Impact of Dataset

The Im2GPS dataset has had a positive impact on computer vision research by enabling the study of global-scale geolocation problems. It has demonstrated that visual features of images can provide meaningful geographic information, which supports applications such as environmental monitoring, mapping, and cultural heritage research.  

At the same time, the dataset also poses risks if misused. Because it contains sensitive geographic information, irresponsible applications could potentially expose private locations or enable surveillance technologies. Researchers using this dataset must therefore ensure that their work respects privacy and is limited to safe, academic, and educational contexts.

### Discussion of Biases

The dataset is not evenly distributed worldwide. Popular tourist destinations, major cities, and regions with active Flickr communities are heavily overrepresented. Rural areas and less photographed regions are underrepresented. This imbalance introduces bias, as models trained on Im2GPS are likely to perform much better in urban or tourist-heavy regions than in sparsely photographed locations. These biases are inherent in the source platform (Flickr) and were not fully corrected in the dataset.

### Other Known Limitations

Beyond geographic imbalance, the dataset has other limitations:  
- The quality of the images varies greatly, ranging from high-resolution scenic photos to low-quality or noisy snapshots.  
- GPS metadata accuracy depends on the devices used by Flickr users and may not always be precise.  
- Despite its scale, the dataset is not exhaustive. The authors estimate that it averages only **0.0435 photos per square kilometer of Earth’s land area**, leaving many parts of the world poorly represented.  

---

## Additional Information

### Dataset Curators

The dataset was created by **James Hays** (Carnegie Mellon University at the time, now Brown University) and **Alexei A. Efros** (Carnegie Mellon University at the time, now UC Berkeley). Funding support came from the **National Science Foundation (NSF)**, and computing resources were provided by **Intel Research Pittsburgh**. Flickr and Yahoo! contributed by hosting the photo platform used for data collection.

### Licensing Information

The dataset does not have a single unified license. Each image retains the license chosen by its original Flickr uploader, many of which are **Creative Commons**. For this reason, the dataset is provided strictly for **academic research purposes only**. Users must comply with the individual photo licenses when displaying or redistributing images.

### Citation Information

If you use this dataset in your research, please cite the original paper:

```bibtex
@inproceedings{hays2008im2gps,
  title={IM2GPS: estimating geographic information from a single image},
  author={Hays, James and Efros, Alexei A},
  booktitle={2008 IEEE Conference on Computer Vision and Pattern Recognition},
  pages={1--8},
  year={2008},
  organization={IEEE}
}
```

### Contributions

The dataset was made possible thanks to the contributions of Flickr and Yahoo! (for hosting the photos), Intel Research Pittsburgh (for computing support), and the National Science Foundation (for funding).

### Benchmark and Auxiliary Resources

- **Flickr download scripts (“Flickr code”):** http://graphics.cs.cmu.edu/projects/im2gps/flickr_code.html  
- **Additional test/utility packs provided by the authors:**  
  - `gps_query_imgs.zip`  
  - `2k_random_test.zip`  
  - `human_geolocation_test.zip`  
  - `geo_uniform_test.zip`  

These packs and scripts were shared on the official project pages to facilitate reproducing experiments and running example queries; photos themselves must be downloaded from Flickr according to each image’s license.
