# Masked Unsupervised Self-training for Zero-shot Image Classification 
This is the PyTorch code of the [MUST paper](https://arxiv.org/abs/2206.02967). The repository supports finetuning a CLIP model on unlabeled images from a target domain.

### Requirements
* pytorch 1.10.0
* timm 0.4.12
* tensorboardX
* ftfy

### Dataset Setup
Dataset paths are stored in [dataset_catalog.json](https://github.com/salesforce/MUST/blob/main/dataset_catalog.json), which need to be modified to local paths. The imagenet dataset follows the standard folder structure. For other datasets, please refer to the scrips from [VISSL](https://github.com/facebookresearch/vissl/tree/main/extra_scripts/datasets) to download and prepare. CLIP's labels and prompt templates are stored in [classes.json](https://github.com/salesforce/MUST/blob/main/classes.json) and [templates.json](https://github.com/salesforce/MUST/blob/main/templates.json).

### Training
Run the following code on 16 A100 GPUs:
<pre>python -m torch.distributed.run --nproc_per_node=16 train.py --dataset [name_of_dataset] --clip_model ViT-B/16 </pre>

### Results
ViT-B/16:
Method | ImageNet | SUN397 | Food101 | GTSRB | DTD | UCF101 
--- | :---: | :---: | :---: | :---: | :---: | :---:
CLIP | 68.3 | 64.4 | 88.7 | 43.4 | 44.7 | 68.8
MUST | 77.7 | 71.8 | 92.7 | 65.5 | 54.1 | 81.1 

ViT-L/14:
Method | ImageNet | SUN397 | Food101 | GTSRB | DTD | UCF101 
--- | :---: | :---: | :---: | :---: | :---: | :---:
CLIP | 75.5 | 67.4 | 92.9 | 50.6 | 55.4 | 77.0
MUST | 82.1 | 74.6 | 95.3 | 68.7 | 62.6 | 85.7 

### Citation
<pre>
@inproceedings{li2022masked,
      title={Masked Unsupervised Self-training for Label-Free Image Classification}, 
      author={Junnan Li and Silvio Savarese and Steven C. H. Hoi},
      year={2023},
      booktitle={ICLR},
}
</pre>
