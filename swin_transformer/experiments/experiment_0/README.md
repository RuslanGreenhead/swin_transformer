# Experiment #0
Swin-T classification on ImageNet-1k - vanilla training (--> Overfitting scenario)
Single GPU used. The configuration of experiment is described below:

* **num of GPUs**: 1
* **num of epochs**: 50 (no lr scheduling)
* **criterion**: nn.CrossEntropyLoss(reductioin='sum')
* **optimizer**: SGD, lr=0.001
* **batch Size**: 128
* **weight Initialization**: --default--
* **regularization**: NO