# Pix2Pix Generative Adversarial Network (GAN)

**Pix2Pix GAN** is a deep learning project that focuses on image-to-image translation tasks. This repository contains the code and resources for training and using a Pix2Pix model to convert images from one domain to another.

## Overview

The Pix2Pix GAN model is a powerful architecture for various image translation tasks. It learns to map input images from one domain to output images in another domain. The model achieves this translation by training on a dataset containing paired images from both domains.

## Features

- **Image-to-Image Translation:** Transform images from one domain to another.
- **Conditional GAN:** Utilizes conditional adversarial networks for improved results.
- **High-Quality Outputs:** Generate high-quality translated images.

## Results

Here are some sample results obtained using our trained Pix2Pix GAN model:

*Input Image -> Label Image -> Generated Image*
![Result 1](results/res1.png)<br>
*Input Image -> Label Image -> Generated Image*
![Result 2](results/res2.png)



## Project Structure

├── data/                 # Data directory
│   ├── dataset.py        # Dataset handling
│   ├── augmentation.py   # Data augmentation functions
│   └── facades/          # Facades dataset directory
│       ├── train/        # Training data
│       ├── val/          # Validation data
│       └── test/         # Test data
├── models/               # Models directory
│   ├── discriminator.py  # Discriminator model
│   ├── generator.py      # Generator model
├── generated/            # Generated images (output)
├── input/                # Input images
├── label/                # Label images
├── utilities/            # Utilities directory
│   ├── config.py         # Configuration settings
├── main.py               # Main script for training and testing
├── requirements.txt      # Python dependencies
├── README.md             # Project README (you are here)
├── Dockerfile            # Dockerfile for containerization
└── .gitignore            # Git ignore file

## Training
 -Clone repo 
 -install requirements - pip -r install requirements.txt
 -run python main.py

## Acknowledgments

- This project is inspired by the Pix2Pix GAN paper by [Isola et al.](https://arxiv.org/abs/1611.07004).
