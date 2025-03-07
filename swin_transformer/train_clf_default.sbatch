#!/bin/bash
#SBATCH --job-name="SwinT_clf_imagenet"     # Название задачи 
#SBATCH --error=task-%j.err                 # Файл для вывода ошибок 
#SBATCH --output=task-%j.log                # Файл для вывода результатов
#SBATCH --mail-user=ramolokanov@edu.hse.ru
#SBATCH --mail-type=ALL 
#SBATCH --time=72:00:00                     # Максимальное время выполнения 
#SBATCH --cpus-per-task=4                   # Количество CPU на одну задачу
#SBATCH --nodes=1                           # Требуемое кол-во узлов 
#SBATCH --gpus=0                            # Требуемое кол-во GPU

module load Python/Google_Colab_GPU_2024        # Загрузка модуля Anaconda 
python train_clf_imagenet.py                    # Выполнение расчёта