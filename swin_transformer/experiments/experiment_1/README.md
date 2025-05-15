# Experiment #1
Training Swin-T model on ImageNet-1k classification. Full-scale training. The configuration of experiment is described below:

* **num of GPUs**: 4
* **num of epochs**: 300 (20 - warmup)
* **criterion**: SoftTargetCrossEntropy
* **optimizer**: AdamW, lr=0.001 (initial), weight_decay=0.05
* **batch Size**: 1024 (256 at each GPU)
* **weight Initialization**: --original--
* **regularization**: --original--