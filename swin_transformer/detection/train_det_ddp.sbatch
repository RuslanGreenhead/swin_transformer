#!/bin/bash
#SBATCH --job-name="SSD_ResNet_det_COCO"    # Название задачи 
#SBATCH --error=task-%j.err                 # Файл для вывода ошибок 
#SBATCH --output=task-%j.log                # Файл для вывода результатов
#SBATCH --mail-user=ramolokanov@edu.hse.ru
#SBATCH --mail-type=ALL 
#SBATCH --time=2:00:00                     # Макс. время выполнения -> 1 сутки 
#SBATCH --cpus-per-task=16                  # 4 gpus -> 16 cpus; 2 gpus - 8 cpus (maintain 1/4 ratio)
#SBATCH --nodes=1                           # Требуемое кол-во узлов 
#SBATCH --gpus=4                            # Требуемое кол-во GPU

module load Python/Google_Colab_GPU_2024        # Загрузка модуля
export NGPUS=4

torchrun --nproc_per_node=$NGPUS training.py        # запуск в параллельном режиме
