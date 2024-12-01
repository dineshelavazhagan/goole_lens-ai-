# goole_lens-ai-

# DeepFashion Similarity Search

## Overview

This repository provides tools and scripts to preprocess the **DeepFashion** dataset for similarity search tasks. The dataset includes images of men's and women's apparel, often featuring models wearing the clothing items. The preprocessing steps isolate upper garments, preparing the data for various similarity search implementations. I chose the deep fashion dataset due to my past experience with fashion searches and matches with VTON solutions; this may be the right dataset to run some evaluations on
model files and the images are huge and are not linked with this repository

## Dataset

The **DeepFashion** dataset is a comprehensive benchmark for fashion-related tasks, containing:

- **Images**: High-resolution images of clothing items worn by models.
- **Labels**: Detailed annotations for different apparel categories.
- **Segmentation Masks**: Pixel-wise masks identifying apparel regions in images.

## Preprocessing Steps

Follow the steps below to prepare the DeepFashion dataset for similarity search:

### Step 1: Download the DeepFashion Dataset

1. **Download the Dataset**:
   - Visit the [DeepFashion Dataset Page](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html) and download the dataset, including:
     - **Images**
     - **Labels**
     - **Segmentation Masks**

2. **Organize the Dataset Directory**:
   - Extract the downloaded files and organize them into the following directory structure:

     ```
     DeepFashion/
     ├── images/
     ├── labels/
     └── masks/
     ```


### Step 2: Extract Clothes Masks

Run the `extract_clothes.py` script to generate masks isolating apparel regions from the images.

```bash
python extract_clothes.py

```

### Step 2: filter tops alone
``` bash

python extracts_tops.py
```

### Step 2: label separation
``` bash

python label_sep.py
```
**make sure the Dataset Directory looks similar to this structure**:
   - Extract the downloaded files and organize them into the following directory structure:

     ```
     labels_sep/
     ├── tops/
        ├── MEN/
           ├── Denim/image1.......jpg
           ├── shirts_polo/image1.......jpg
        └── WOMEN/
           ├── Denim/image1.......jpg
           ├── sweters/image1.......jpg
     ```


## 1. FEATURE EXTRACTION

Extracts feature, indexes with LSH, and retrieves similar images.

## Evaluation Metrics

| Model                     | Precision | Recall | F1 Score |
|---------------------------|-----------|--------|----------|
| Feature Extraction + LSH  | 0.72      | 0.68   | 0.70     |


## Usage

1. **Install Dependencies**:
   ```bash
   pip install torch torchvision numpy matplotlib tqdm pillow
   ```
1. **Run code**:
   ```bash
   python fe.py
   ```
   you can make changes for dataset_dir for overall search where it searches all images instead of label-separated
   change query_image_path for different query images to try

**This approach does not give the desired output BUT the FIASS index gives far better results compared to LSH**




## 2. CLIP

Generates CLIP embeddings, builds FAISS index, retrieves, and visualizes similar images efficiently.

## Evaluation Metrics

| Model        | Precision | Recall | F1 Score |
|--------------|-----------|--------|----------|
| CLIP + FAISS | 0.95      | 0.90   | 0.92     |

 **This approach Gave higher similarity results and expected outputs and had higher accuracy compared to other approaches**

1. **Run code**:
   ```bash
   python clip_similarity_search.py
   ```
   you can make changes for dataset_dir for overall search where it searches all images instead of label-separated
   change query_image for different query images to try
   

## 3. Autoencoders:

Trains a convolutional autoencoder, extracts latent vectors, indexes with FAISS, and retrieves and visualizes similar images.

## Evaluation Metrics

| Model                 | Precision | Recall | F1 Score |
|-----------------------|-----------|--------|----------|
| ConvAutoencoder FAISS | 0.62      | 0.68   | 0.64     |

1. **Run code**:
   ```bash
   python auto_encode_similarity_search.py
   ```
**This approach has very low accuracy, further model training and better preparation of datasets can yield better results**

## 4. Triplet-loss network:

Trains a ResNet18-based triplet loss network to learn image embeddings for similarity search.

## Evaluation Metrics

| Model                  | Precision | Recall | F1 Score |
|------------------------|-----------|--------|----------|
| Triplet Loss Network   | 0.60      | 0.55   | 0.57     |


1. **Run code**:
   ```bash
   python triplet_loss.py
   ```
**This approach has very low accuracy, further model training and better preparation of datasets can yield better results**


# Best Similarity Search

![alt text](https://github.com/dineshelavazhagan/goole_lens-ai-/blob/main/Figure_1.png)

Compared to all other models CLIP gave perfect results on all similarity searches


