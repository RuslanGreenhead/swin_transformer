# Swin Transformer

This project is an implementation of the paper **"Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"**. The Swin Transformer serves as a general-purpose backbone for computer vision tasks, effectively addressing the challenges of adapting Transformer models from language to vision.

### Model Architecture:

![Swin Transformer Architecture](https://amaarora.github.io/images/swin-transformer.png)

### Repo structure:

 ```
📦swin_transformer
 ┣ 📂experiments
 ┃ ┗ 📂experiment_0                        # Swin-T -> single testing epoch
 ┃ ┃ ┣ 📜basic_model.pth
 ┃ ┃ ┣ 📜task-2403241.log
 ┃ ┃ ┗ 📜training_output.pkl
 ┃ 
 ┣ 📂notebooks
 ┃ ┣ 📜ImageNet_classification.ipynb       # debugging ImageNet proceeding
 ┃ ┗ 📜playground.ipynb                    # notebook with scetching & testing the modules 
 ┃
 ┣ 📜model.py                              # all the building blocks of the network
 ┣ 📜train_clf_imagenet.py                 # training code -> ImageNet 1K [classification]
 ┣ 📜train_clf_imagenet.sbatch             # cluster task  -> ImageNet 1K [classification]
 ┗ 📜__init__.py
 ```

### Links & Sources:

* [Original paper](https://arxiv.org/abs/2103.14030)
* [Official implementation by Microsoft](https://github.com/microsoft/Swin-Transformer)