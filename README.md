# Breast Cell Classification with Fluorescence Microscopy

## 🧬 Project Overview

This project focuses on the classification of breast cells as **benign** or **malignant** using images acquired through **fluorescence microscopy** — one of the most advanced imaging techniques in cellular analysis. Fluorescence microscopy uses fluorescent markers to highlight specific cellular structures, allowing detailed visualization of features such as the nucleus and membrane.

These images typically contain **multiple optical channels**, each corresponding to a different fluorescent label. This enables simultaneous observation of various cellular components, which is particularly valuable in cancer research and diagnostics, as it helps detect subtle morphological and molecular differences between healthy and pathological cells.

---

## 🎓 Academic Context

This project was developed as part of a **mini contest** organized by the *University of Naples Federico II* for the **Machine Learning** course.  
The objective was to design and train a **deep neural network** for binary classification of breast cells, aiming to maximize the prediction **Accuracy** on a given test set.

---

## 🎯 Objective

Each participant was tasked with developing one or more **prediction models** using **data analysis** and **machine learning techniques** to classify each breast cell as benign or malignant.  
The primary evaluation metric is **Accuracy**.

---

## 🔬 Dataset Description

The dataset employed in this study is provided by **DICMAPI** and comprises **fluorescence microscopy images** acquired from two human breast cell lines:

- **MCF10a**: representing non-tumorigenic (healthy) epithelial cells  
- **MCF7**: a widely used model of malignant breast cancer cells

The dataset is divided as follows:

- **Training set**: 239 acquisitions  
- **Test set**: 60 acquisitions

Since each acquisition contains **multiple cells**, a preprocessing algorithm is used to segment and extract **individual cells**, treating each one as a separate image. For each extracted cell, **three distinct optical channels** are provided and stored as `.npy` files:

| Channel       | Description                         |
|---------------|-------------------------------------|
| **M (Membrane)**     | Highlights the cell membrane structure |
| **N (Nucleus)**      | Fluorescent signal from the nucleus    |
| **T (Transmission)** | Brightfield image (non-fluorescent)    |

Each image is uniquely identified by an ID encoding the **acquisition number** and the **cell index**.  
For example, `000_1` refers to the first cell extracted from acquisition `000`.

- 📦 **Training images**: 6,798  
- 📦 **Test images**: 1,721  

The classification task is **binary**:

- **0** → Benign (healthy)  
- **1** → Malignant (cancerous)

---

## 📊 Evaluation

The evaluation metric for this competition is **Accuracy**.

Accuracy is defined as the percentage of correctly classified instances with respect to the total number of evaluated samples:
Accuracy = (True Positives + True Negatives) / Total Evaluated Instances
