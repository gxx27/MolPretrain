# Pre-training of heterogeneous graph neural networks for molecular graphs

## Dataset
Dataset are available at the Google Drive link:https://drive.google.com/drive/folders/1FPmZilTbS5qnR-ehOZxn0aSf6y5gg8Go

## KPGT
First, you need to download the dataset and pre-trained model in the file KPGT/

and then build the conda environment
```powershell
conda env create
conda activate KPGT
```

For pre-training, run the following command
```powershell
cd scripts
./pretrain.sh
```
to do the pretraining

For fine-tuning, run the following command
```powershell
cd scripts
./finetune_classification.sh # downstream task is classification

# or
./finetune_regression.sh # downstream task is regression
```
to do the fine-tuning

You can change the downstream task in the shell files, please choose 5-6 tasks which performs best.

## Files
KPGT: dynamic masking + md_pred + fp_pred + enlarged dataset

rm_both_pred: dynamic masking + enlarged dataset

rm_fp_pred: dynamic masking + md_pred + enlarged dataset

rm_md_pred: dynamic masking + fp_pred + enlarged dataset