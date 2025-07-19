# Swin Transformer

This project is an implementation of the paper **"Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"**. The Swin Transformer serves as a general-purpose backbone for computer vision tasks, effectively addressing the challenges of adapting Transformer models from language to vision.

### Model Architecture:

![Swin Transformer Architecture](https://amaarora.github.io/images/swin-transformer.png)

### Repo structure:

 ```
📦swin_transformer
 ┣ 📂configs
 ┃ ┣ 📜detection_ssd_resnet50.yaml
 ┃ ┗ 📜SwinT_clf_300e_default.yaml
 ┃
 ┣ 📂detection
 ┃ ┣ 📜backbones.py
 ┃ ┣ 📜data.py
 ┃ ┣ 📜ssd.py
 ┃ ┣ 📜test.py
 ┃ ┣ 📜test_detection.sbatch
 ┃ ┣ 📜training.py
 ┃ ┣ 📜train_det_ddp.sbatch
 ┃ ┗ 📜utils.py
 ┃ 
 ┣ 📂experiments
 ┃ ┣ 📂experiment_0
 ┃ ┣ 📂experiment_1
 ┃ ┗ 📂experiment_FAIL
 ┃
 ┣ 📂notebooks
 ┃ ┣ 📜ImageNet_classification.ipynb
 ┃ ┗ 📜playground.ipynb
 ┃ 
 ┣ 📂saved_weights
 ┃ 
 ┣ 📜data.py
 ┣ 📜model.py
 ┣ 📜training.py
 ┣ 📜train_clf_ddp.sbatch
 ┣ 📜train_clf_imagenet.py
 ┗ 📜__init__.py
 ```

### Links & Sources:

* [Original paper](https://arxiv.org/abs/2103.14030)
* [Official implementation by Microsoft](https://github.com/microsoft/Swin-Transformer)