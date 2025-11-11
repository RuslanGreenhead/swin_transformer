#!/bin/bash
#SBATCH --job-name="SwinT_duff_mnist"       # Название задачи 
#SBATCH --error=task-%j.err                 # Файл для вывода ошибок 
#SBATCH --output=task-%j.log                # Файл для вывода результатов
#SBATCH --mail-user=ramolokanov@edu.hse.ru
#SBATCH --mail-type=ALL 
#SBATCH --time=12:00:00                     # Макс. время выполнения
#SBATCH --cpus-per-task=8                   # 4 gpus -> 16 cpus; 2 gpus - 8 cpus (maintain 1/4 ratio)
#SBATCH --nodes=1                           # Требуемое кол-во узлов 
#SBATCH --gpus=2                            # Требуемое кол-во GPU

# variables for synchronized validation issues
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_IB_TIMEOUT=22
# export NCCL_DEBUG=INFO

module load Python/Google_Colab_GPU_2024        # Загрузка модуля
export NGPUS=2

torchrun --nproc_per_node=$NGPUS training.py        # запуск в параллельном режиме