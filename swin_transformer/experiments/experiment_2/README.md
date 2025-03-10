# Experiment #2
Training Swin-T model on ImageNet-1k classification. First successful vanilla-trial training. The configuration of experiment is described below:

* **num of GPUs**: 1
* **num of epochs**: 10
* **criterion**: nn.CrossEntropyLoss(reduction='mean')
* **optimizer**: SGD(model.parameters(), lr=0.001)
* **batch Size**: 128
* **weight Initialization**: --default--
* **regularization**: --default--