#!/bin/bash
#SBATCH -o %j.out 
#SBATCH -e %j.out 
#SBATCH -J EXPO # 作业名指定为test
#SBATCH -p si 
#SBATCH --nodes=1             # 申请一个节点
#SBATCH --gres=gpu:1		#分配的gpu数量
#SBATCH --cpus-per-task=10 # 一个任务需要分配的CPU核心数为5
#SBATCH --time=999:00:00
# 需要执行的指令

models=("GRU4Rec" "FMLPRecModel" "SASRec"  "LightSANs" "Mamba4Rec" "Linrec")
datasets=("Beauty")

batch_size=1
current_batch=()

# 循环运行命令
for model in "${models[@]}"
do
  for dataset in "${datasets[@]}"
  do
    echo "Running for model $model on dataset $dataset..."
    CUDA_VISIBLE_DEVICES=2 python3 -u run_finetune_full.py \
    --data_name="$dataset" \
    --ckp=0 \
    --hidden_size=64 \
    --backbone="$model" > "res/${dataset}-${model}.txt" 
    # 将当前进程的 PID 添加到批次数组中
    current_batch+=($!)

    # 如果当前批次已满，等待所有进程完成并清空批次数组
    if [ ${#current_batch[@]} -eq $batch_size ]; then
      wait "${current_batch[@]}"
      current_batch=()
    fi
  done
done

# 等待最后一批进程完成
if [ ${#current_batch[@]} -ne 0 ]; then
  wait "${current_batch[@]}"
fi

echo "All runs completed."
