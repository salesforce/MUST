# Masked Unsupervised Self-training for Zero-shot Image Classification 
This is the PyTorch code of the [MUST paper](). The repository supports finetuning a CLIP model on unlabeled images from a target domain.

### Requirements
* pytorch 1.10.0
* timm 0.4.12
* tensorboardX
* ftfy

### Dataset Setup
Dataset paths are stored in [dataset_catalog.json](https://github.com/salesforce/MUST/blob/main/dataset_catalog.json), which need to be modified to local paths. The imagenet dataset follows the standard folder structure. Please refer to the scrips from [VISSL](https://github.com/facebookresearch/vissl/tree/main/extra_scripts/datasets) to prepare other datasets. CLIP's labels and prompt templates are stored in [classes.json](https://github.com/salesforce/MUST/blob/main/classes.json) and [templates.json](https://github.com/salesforce/MUST/blob/main/templates.json).

### Training
Run the following code on 16 A100 GPUs:
<pre>python -m torch.distributed.run --nproc_per_node=16 train.py --dataset [name_of_dataset] --clip_model ViT-B/16 </pre>

### Citation
