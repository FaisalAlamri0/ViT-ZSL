# ViT-ZSL

[PyTorch](https://pytorch.org/) | [Arxiv]()

PyTorch implementation of our ViT-ZSL model for zero-shot learning:  
[Multi-Headed Self-Attention via Vision Transformer for Zero-Shot Learning]()  
[Faisal Alamri](), [Anjan Dutta](https://sites.google.com/site/2adutta/)   
[IMVIP, 2021](https://imvipconference.github.io/)

![](figs/ViT-ZSL%20Architecture.jpg)

## Abstract
Zero-Shot Learning (ZSL) aims to recognise unseen object classes, which are not observed during the trainingphase.  The existing body of works on ZSL mostly relies on pretrained visual features and lacks the explicitattribute localisation mechanism on images. In this work, we propose an attention-based model in the problemsettings of ZSL to learn attributes useful for unseen class recognition. Our method uses an attention mechanismadapted from Vision Transformer to capture and learn discriminative attributes by splitting images into smallpatches.   We conduct experiments on three popular ZSL benchmarks (i.e.,  AWA2,  CUB and SUN) and setnew state-of-the-art harmonic mean results on all the three datasets, which illustrate the effectiveness of ourproposed method.


## Usage:
#### 1) Download the datasets
Follow the instructions provided in [data/Dataset_Instruction.txt](data/Datasets_Instruction.txt)


#### 2) Create a conda environment:
Refer to: [Conda Environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for more information. 
```
# conda create -n {ENVNAME} python=3.6
conda create -n ViT_ZSL python=3.6

# Activate the environment: conda activate {ENVNAME}
conda activate ViT_ZSL
```
#### 3) Required libraries :
This is a [PyTorch](https://pytorch.org/get-started/locally/) implementation
```
pip install -r requirements.txt 

# PyTorch
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
```
#### 4) Train (and test) the model
open [ViT_ZSL.ipynb](ViT_ZSL.ipynb)
```
jupyter notebook ViT_ZSL.ipynb
```


## External sources:

- [Timm](https://pypi.org/project/timm/)
- [Transformer](https://github.com/huggingface/transformers)
- [ViT](https://github.com/google-research/vision_transformer)
- [Vision_Transformer_Tutorial](https://colab.research.google.com/github/hirotomusiker/schwert_colab_data_storage/blob/master/notebook/Vision_Transformer_Tutorial.ipynb#scrollTo=3f7gQ89cvAnv)


## Further questions:
Please do read [our paper]().
If you still require any further information, please feel free to contact us at our emails. 

## Citation:
If you use ViT-ZSL in your research, please use the following BibTeX entry.
```
@InProceedings{Alamri2021ViTZSL,
  author    = {Faisal Alamri and Anjan Dutta},
  title     = {Multi-Headed Self-Attention via Vision Transformer for Zero-Shot Learning},
  booktitle = {IMVIP},
  year      = {2021}
}
```

## Authors
* [Faisal Alamri]() ([@FaisalAlamri](https://github.com/FaisalAlamri0))
* [Anjan Dutta](https://sites.google.com/site/2adutta/) ([@AnjanDutta](https://github.com/AnjanDutta))

