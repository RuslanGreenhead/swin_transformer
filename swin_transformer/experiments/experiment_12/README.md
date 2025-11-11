# Experiment #12
SwinUNet diffusion on CIFAR-10 dataset. Bicubic iterpolation of CIFAR-10 imahes to 224x224. 
Time passed only before the first basic layer. Condition tensor is acquired by expandng (cond_dim, 1, 1) to 
(cond_dim, 224, 224).

**Model & data configuration:**
* **input resolution**: 224x224
* **n_timesteps**: 300
* **time_emb_dim**: (?)
* **condition_emb_dim**: 4
* **condition_dim**: 32
* **augmentations**: random resized crops + horzontal flips

**Training setup:**
* **num of GPUs**: 2
* **batch size**: 128 (64 per GPU)
* **num of epochs**: 75 (3 - warmup)
* **optimizer**: Adam, lr=0.0001 (initial), weight_decay=0.05 (except BNs, biases, pos. encoding)
* **scheduling**: cosine annealing
* **weight Initialization**: --vanilla--