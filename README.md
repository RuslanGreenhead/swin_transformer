# Swin Transformer

This project is an implementation of the paper **"Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"**. The Swin Transformer serves as a general-purpose backbone for computer vision tasks, effectively addressing the challenges of adapting Transformer models from language to vision.

### Model Architecture:

![Swin Transformer Architecture](https://amaarora.github.io/images/swin-transformer.png)

### Repo structure:

 ```
📦swin_transformer
 ┣ 📂experiments
 ┃ ┗ 📂experiment_0                        # Swin-T -> single testing epoch
 ┃ ┗ 📂experiment_1                        # Swin-T -> training over 30 epochs (2 GPU)
 ┃ ┗ 📂experiment_2                        # Swin-T -> training over 10 epochs (2 GPU)
 ┃ ┗ 📂experiment_3                        # Swin-T -> training over 50 epochs (same setup as experiment_2)
 ┃ ┗ 📂experiment_4                        # Swin-T -> full-scale training (300(20 - warmup) epochs) (2 GPU)
 ┃ 
 ┣ 📂notebooks
 ┃ ┣ 📜ImageNet_classification.ipynb       # debugging ImageNet proceeding
 ┃ ┗ 📜playground.ipynb                    # notebook with scetching & testing the modules 
 ┃
 ┣ 📜model.py                              # all the building blocks of the network
 ┣ 📜data.py                               # data processing issues
 ┣ 📜training.py                           # training code -> ImageNet 1K [classification]
 ┣ 📜train_clf_ddp.sbatch                  # cluster task -> ImageNet 1K [classification]
 ┗ 📜__init__.py
 ```

### Links & Sources:

* [Original paper](https://arxiv.org/abs/2103.14030)
* [Official implementation by Microsoft](https://github.com/microsoft/Swin-Transformer)