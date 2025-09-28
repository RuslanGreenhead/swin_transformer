# Swin Transformer

This project is an implementation of the paper **"Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"**. The Swin Transformer serves as a general-purpose backbone for computer vision tasks, effectively addressing the challenges of adapting Transformer models from language to vision.

### Model Architecture:

![Swin Transformer Architecture](https://amaarora.github.io/images/swin-transformer.png)

### Detection:

| necks▶<br>backbones▼  | neck_1 | neck_2 |
|-----------------------|--------|--------|
| model_A               |        |        |
| model_B               |        |        |

### Repo structure:

 ```
📦swin_transformer
 ┣ 📂configs                          # training configurations
 ┃ ┣ 📜SwinT_clf_300e_default.yaml
 ┃ ┣ 📜detection_ssd_SwinT.yaml
 ┃ ┗ 📜detection_ssd_resnet50.yaml
 ┃ 
 ┣ 📂detection
 ┃ ┣ 📜backbones.py
 ┃ ┣ 📜data.py
 ┃ ┣ 📜necks.py
 ┃ ┣ 📜ssd.py
 ┃ ┣ 📜test.py                        # source for full validation pipeline
 ┃ ┣ 📝test_detection.sbatch          # slurm script -> test.py (no DDP)
 ┃ ┣ 📝train_det_ddp.sbatch           # slurm script -> training.py (DDP)
 ┃ ┣ 📜training.py                    # source for training
 ┃ ┗ 📜utils.py
 ┃
 ┣ 📂experiments
 ┃ ┣ 📂experiment_0                   # trial run
 ┃ ┣ 📂experiment_1                   # ImageNet classification -> SwinT
 ┃ ┣ 📂experiment_2                   # SSD detection -> ResNet (B)
 ┃ ┣ 📂experiment_3                   # SSD detection -> ResNet (B) + FPN
 ┃ ┣ 📂experiment_4                   # SSD detection -> ResNet (B) + PAN
 ┃ ┣ 📂experiment_5                   # SSD detection -> ResNet (B) + DenseFPN
 ┃ ┣ 📂experiment_6                   # SSD detection -> SwinT (A) (short scheduling)
 ┃ ┣ 📂experiment_7                   # SSD detection -> SwinT (B) (short scheduling)
 ┃ ┣ 📂experiment_8                   # SSD detection -> SwinT (B)
 ┃ ┗ 📂experiment_9                   # SSD detection -> SwinT (B) + FPN  
 ┃
 ┣ 📂notebooks                        # some old colab drafts 
 ┃ ┣ 📜ImageNet_classification.ipynb
 ┃ ┗ 📜playground.ipynb
 ┃
 ┣ 📂saved_weights
 ┃ ┣ 💾SwinT_statedict.pth            # stored locally 
 ┃ ┗ 💾resnet50_statedict.pth         # stored locally
 ┃
 ┣ 📜data.py
 ┣ 📜model.py                         # Swin Transformer implementation
 ┣ 📝train_clf_ddp.sbatch.            # slurm script -> training.py (DDP)
 ┣ 📝train_clf_default.sbatch
 ┣ 📜train_clf_imagenet.py
 ┗ 📜training.py                      # source for classification training
 ```

### Links & Sources:

* [Original paper](https://arxiv.org/abs/2103.14030)
* [Official implementation by Microsoft](https://github.com/microsoft/Swin-Transformer)