# Pre-training of heterogeneous graph neural networks for molecular graphs

## Dataset
Plase use gdown to download the pre-training datasets
```shell
gdown 1Vo8X0MN_Ni7H1HJRR4NKrV1lzQN4A_fu # for chembl dataset
gdown 14YJIlgHEu4Qrp1asYxgzZNCu3GZxi2l5 # for pubchem dataset
```

## KPGT
First, you need to download the dataset and pre-trained model in the file KPGT/

and then build the conda environment
```shell
conda env create
conda activate KPGT
```

For pre-training, run the following command
```shell
cd scripts
./pretrain.sh
```
to do the pretraining

For fine-tuning, run the following command
```shell
cd scripts
./finetune_classification.sh # downstream task is classification

# or
./finetune_regression.sh # downstream task is regression
```
to do the fine-tuning

You can change the downstream task in the shell files.