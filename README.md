# Vision Transformer (ViT) for Cats vs Dogs Classification 🐱🐶

This repository contains an implementation of a **Vision Transformer (ViT)** model fine-tuned for the **Cats vs Dogs classification** task using **PyTorch** and **timm**.

## 🚀 Features
- Uses [timm](https://github.com/huggingface/pytorch-image-models) to load a pretrained ViT model.
- Fine-tunes on a **binary classification task** (Cats vs Dogs).
- Data augmentation and preprocessing with `timm.data`.
- Training loop with evaluation metrics:
  - Accuracy
  - F1-Score
- Visualization of training progress.

## 📂 Project Structure
```

ViT_CvsD.ipynb   # Main Jupyter notebook for training and evaluation

````

## ⚙️ Requirements
Install dependencies with:
```bash
pip install torch torchvision timm torchmetrics matplotlib tqdm numpy
````

## 🏋️‍♂️ Training

Open the notebook and run all cells to train the model:

```bash
jupyter notebook ViT_CvsD.ipynb
```

## 📊 Results

* Model: `vit_small_patch16_224` (pretrained on ImageNet)
* Evaluated using Accuracy and F1-Score
* Supports GPU training (CUDA)

## 📸 Example Predictions

After training, the model can predict whether an image is a **cat** or a **dog**.

## 🙌 Acknowledgements

* [timm](https://github.com/huggingface/pytorch-image-models) for pretrained models
* [PyTorch](https://pytorch.org/) for deep learning framework
