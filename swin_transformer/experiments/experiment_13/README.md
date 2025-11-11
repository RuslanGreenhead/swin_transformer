# Experiment #13
SwinUNet diffusion on CIFAR-10 dataset. Bicubic iterpolation of CIFAR-10 images to 224x224. 
Time passed only before the first basic layer. 
Condition tensor is acquired as following: 
    get class embedding -> pass it through MLP and reshape the result to (-1, self.condition_dim, 16, 16) ->
    -> bilinear interpolation of <u>this</u> tensor to 224x224.

**Model & data configuration:**
* **dataset**: CIFAR-10 (3-channeled)
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