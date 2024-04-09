export CUDA_VISIBLE_DEVICES=2,3
n_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
nproc_per_node=$n_devices
n_threads=$((n_devices * 2))

python -u -m torch.distributed.run --nproc_per_node=$nproc_per_node \
    --nnodes=1 \
    --master_port 12312 train_kpgt.py \
    --save_path ../models/ \
    --n_threads $n_threads \
    --n_devices $n_devices \
    --config KPGT-B/768 \
    --n_steps 100000 \
    --pretrain1_path ../../dataset/chembl29 \
    --pretrain2_path ../../dataset/pubchem-10m \
    --pretrain_strategy rm_none_pred