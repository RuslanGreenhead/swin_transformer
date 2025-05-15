# Experiment #2
Training Swin-T model on ImageNet-1k classification. First full-scale training. The configuration of experiment is described below:

* **num of GPUs**: 2
* **num of epochs**: 300 (20 - warmup)
* **criterion**: nn.CrossEntropyLoss(reduction='sum')
* **optimizer**: AdamW, lr=0.001, weight_decay=0.05
* **batch Size**: 512 (256 at each GPU)
* **weight Initialization**: --default--
* **regularization**: --default--