
# BRAIN TUMOR SEGMENTATION WITH CONVOLUTIONAL NEURAL NETWORKS: U-NET BASED APPROACHES

## Description
This repository contains the implementation of various U-Net architectures for brain tumor segmentation as part of a bachelor thesis. The objective of this project is to explore and compare the performance of different U-Net models in segmenting brain tumors from multi-modal Magnetic Resonance Imaging (MRI) scans. The implemented models include:

    1. 2D U-Net
    2. 3D U-Net
    3. Residual U-Net
    4. Attention U-Net
    5. Residual Attention U-Net

## Project Overview
Brain tumor segmentation is a crucial step in the diagnosis and treatment planning process for patients with brain tumors. Accurate and efficient segmentation of brain tumors from MRI scans can provide valuable information for clinicians and researchers. This project aims to develop and evaluate deep learning models based on U-Net architectures to achieve high-quality brain tumor segmentation.

The implemented models are developed in Python using the TensorFlow and Keras libraries. The repository includes pre-processing methods tailored for MRI scans, as well as model training, evaluation, and visualization tools.

## Dataset
The models are trained and evaluated on the Brain Tumor Segmentation 2021 (BraTS) dataset, which consists of multi-modal MRI scans from patients with gliomas.

## Repository Structure
* 'Logs/' : Contains the training/testing/validation logs from training and evaluation of the models. 
* 'Models/' : Contains the h5 files for the models trained. 
* 'architectures/' : Contains the .py files for the five different architectures listed in the description. 
* 'Training_' : Files used to train each of the models. These can be ran from top to bottom.
* 'metrics' : Contains the evaluation metrics used. 
* 'datagenerators' : contains the datagenerators used to generate data while training. The PATH variables in this code must be changed to your own path to the data if the training_ files is to be ran. 

## Getting Started
To get started and to use the code in this repository, follow these steps:

    1. Clone the repository to your local machine
    2. Download the BraTS dataset and place it on your local machine.
    3. Install the required Python packages listed in requirements.txt
    4. Change the PATH variables in the 'datagenerators.py' to the path to the data on your local machine. 
    5. Train and evaluate the models running the 'Training_'.ipynb files from top to bottom. 

## Results
The best performing architecture in this project was the Residual U-Net architecture. Trained on 300 samples, it achieved Dice Similarity Coefficients of 0.8107 (overall), and 0.8917, 0.8693, and 0.8187 for edema, enhancing, and necrotic
regions respectively

An example of model predictions can be viewed in the image below. 

![grouped](https://user-images.githubusercontent.com/93523562/233627559-ed45f608-bdf9-4d8b-9452-d0045cde1126.png)

## Contact
For any questions, comments, or concerns, please feel free to reach out to the repository owner. 
