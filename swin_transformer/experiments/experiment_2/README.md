# Experiment #8
Training SSD with ResNet50 backbone (type A) on COCO dataset. The configuration of experiment is described below:
Difference from Experiment #7 - batch size per GPU (32 -> 64), total batch size (128 -> 256)

**Model & data configuration:**
* **input resolution**: 300x300
* **backbone**: ResNet50Backbone_A (-> 3 intrinsic + 3 auxiliary feature maps)
* **1st feature map rescaling**: YES, trained, init: 20
* **augmentations**: pattern #1 (TORCHVISION_SSD-style)


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