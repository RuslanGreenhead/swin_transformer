# Experiment #4
Training SSD with ResNet50 backbone (type B) on COCO dataset. The configuration of experiment is described below:
Difference from Experiment #3 -> using FPN neck

**Model & data configuration:**
* **input resolution**: 300x300
* **backbone**: ResNet50Backbone_B (-> 1 intrinsic + 5 auxiliary feature maps)
* **1st feature map rescaling**: NO
* **augmentations**: pattern #1 (TORCHVISION_SSD-style)
* **neck**: FPN (Feature Pyramid Network)


**Training setup:**
* **num of GPUs**: 4
* **batch size**: 256 (64 per GPU)
* **num of epochs**: 65 (1 - warmup)
* **optimizer**: SGD, lr=0.0026 (initial), weight_decay=0.0005 (except BNs & biases)
* **scheduling**: 0.1 drop (milestones - 43, 54)
* **weight Initialization**: --vanilla--

**Inference setup:**
* **max_overlap**: 0.45
* **min_score**: 0.05
* **top_k**: 200