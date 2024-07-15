python finetune.py --batch_size 32 \
    --config KG-GCPT \
    --cuda cuda:0 \
    --dataset esol \
    --dataset_type regression \
    --dropout 0 \
    --lr 1e-5 \
    --metric rmse \
    --model_path ../models/cpcontrastive_sub_node_1024.pth \
    --n_epochs 50 \
    --weight_decay 0 \
    --seed 22