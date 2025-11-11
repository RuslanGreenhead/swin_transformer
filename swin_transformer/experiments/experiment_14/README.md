# Experiment #14
SwinUNet diffusion on MNIST dataset. Bicubic iterpolation of MNIST digits to 224x224. 
No pretrained weights, treat images as one-channeled (1, 224, 224).
Time passed before <u>each</u> basic layer of a downward part. (time as (B, 1, 1, emb_dim) is added to x)
Condition tensor is acquired by expandng (cond_dim, 1, 1) to (cond_dim, 224, 224).

**Model & data configuration:**
* **dataset**: MNIST (1-channeled)
* **input resolution**: 224x224
* **n_timesteps**: 500 
* **time_emb_dim**: 128
* **condition_emb_dim**: 32
* **condition_dim**: 4
* **augmentations**: random resized crops + horzontal flips

**Training setup:**
* **num of GPUs**: 2
* **batch size**: 128 (64 per GPU)
* **num of epochs**: 75 (3 - warmup)
* **optimizer**: Adam, lr=0.0001 (initial), weight_decay=0.0001 (except BNs, biases, pos. encoding)
* **scheduling**: cosine annealing
* **weight Initialization**: --vanilla--