# Face recoginition project
This repository is a project for face recognition. 

## Environment Setup
We provide all environment configurations in ``requirements.txt``. To install all packages, you can create a conda environment and install the packages as follows: 
```bash
conda create -n face python=3.8
conda activate face
pip install -r environment.txt
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```
In our experiments, we used NVIDIA CUDA 11.3 on Ubuntu 20.04. Similar CUDA version should also be acceptable with corresponding version control for ``torch`` and ``torchvision``.

## Data processing
Before you start training, you need to run ``data_process.py`` to process the raw data. 

## Training

To train the model from scratch we provide the following model files:
 - ``train_trans_dec.py``: transformer-based model
 - ``train_mixture_dec.py``: mixture-based model
 - ``train_base_sa.py``: original slot-attention
We provide training scripts under ```scripts/train```. Please use the following command and change ``.sh`` file to the model you want to experiment with. Take the transformer-based decoder experiment on Birds as an exmaple, you can run the following:
```bash
$ cd scripts
$ cd train
$ chmod +x trans_dec_birds.sh
$ ./trans_dec_birds.sh
```
## Reload ckpts & test_only

To reload checkpoints and only run inference, we provide the following model files:
 - ``test_trans_dec.py``: transformer-based model
 - ``test_mixture_dec.py``: mixture-based model
 - ``test_base_sa.py``: original slot-attention

Similarly, we provide testing scripts under ```scripts/test```. We provide transformer-based model for real-world datasets (Birds, Dogs, Cars, Flowers) 
and mixture-based model for synthetic datasets(ShapeStacks, ObjectsRoom, ClevrTex, PTR). We provide all checkpoints at this [link](https://drive.google.com/drive/folders/10LmK9JPWsSOcezqd6eLjuzn38VdwkBUf?usp=sharing). Please use the following command and change ``.sh`` file to the model you want to experiment with:
```bash
$ cd scripts
$ cd test
$ chmod +x trans_dec_birds.sh
$ ./trans_dec_birds.sh
```

## Citation
If you find our paper and/or code helpful, please consider citing:
```
@article{jia2022egotaskqa,
    title = {Unsupervised Object-Centric Learning with Bi-Level Optimized Query Slot Attention},
    author = {Jia, Baoxiong and Liu, Yu and Huang, Siyuan},
    journal = {arXiv preprint arXiv:2210.08990},
    year = {2022}
}
```

## Acknowledgement
This code heavily used resources from [SLATE](https://github.com/singhgautam/slate), [SlotAttention](https://github.com/untitled-ai/slot_attention), [GENESISv2](https://github.com/applied-ai-lab/genesis), [DRC](https://github.com/yuPeiyu98/DRC.git), [Multi-Object Datasets](https://github.com/deepmind/multi_object_datasets), [shapestacks](https://github.com/ogroth/shapestacks). We thank the authors for open-sourcing their awesome projects.
