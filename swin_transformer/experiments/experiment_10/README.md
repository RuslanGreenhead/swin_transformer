# Experiment #10
Training SSD with SwinT backbone (type B) on COCO dataset. The configuration of experiment is described below:
Difference from Experiment #8 -> using PAN neck

**Model & data configuration:**
* **input resolution**: 336x336
* **backbone**: SwinTBackbone_B (-> 1 intrinsic + 5 auxiliary feature maps)
* **1st feature map rescaling**: NO
* **augmentations**: pattern #1 (TORCHVISION_SSD-style)
* **neck**: PAN (Path Aggregating Network)

**Training setup:**
* **num of GPUs**: 4
* **batch size**: 256 (64 per GPU)
* **num of epochs**: 52 (2 - warmup)
* **optimizer**: Adam, lr=0.0001 (initial), weight_decay=0.05 (except BNs, biases, pos. encoding)
* **scheduling**: 0.1 drop (milestones - 32, 42, 49)
* **weight Initialization**: --vanilla--

**Inference setup:**
* **max_overlap**: 0.45
* **min_score**: 0.05
* **top_k**: 200