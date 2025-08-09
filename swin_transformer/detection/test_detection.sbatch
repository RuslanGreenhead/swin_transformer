#!/bin/bash
#SBATCH --job-name="SwinT_det_COCO"     # Название задачи 
#SBATCH --error=task-%j.err                 # Файл для вывода ошибок 
#SBATCH --output=task-%j.log                # Файл для вывода результатов
#SBATCH --mail-user=ramolokanov@edu.hse.ru
#SBATCH --mail-type=ALL 
#SBATCH --time=1:00:00                    # Макс. время выполнения -> 25 суток 
#SBATCH --cpus-per-task=2                  # 4 gpus -> 16 cpus; 2 gpus - 8 cpus (maintain 1/4 ratio)
#SBATCH --nodes=1                           # Требуемое кол-во узлов 
#SBATCH --gpus=1                            # Требуемое кол-во GPU

module load Python/Google_Colab_GPU_2024        # Загрузка модуля

python test.py                                # запуск в стандартном режиме