# Face recoginition project
This repository is a project for face recognition. 

## Environment Setup
We provide all environment configurations in ``requirements.txt``. To install all packages, you can create a conda environment and install the packages as follows: 
```bash
conda create -n face python=3.8
conda activate face
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
pip install dlib
pip install face_recognition
```
In our experiments, we used NVIDIA CUDA 11.3 on Ubuntu 20.04. Similar CUDA version should also be acceptable with corresponding version control for ``torch`` and ``torchvision``.

## Data processing
Before you start training, you need to run ``data_process.py`` to process the raw data. Remember to change the ``DATA_ROOT`` to your own path. 

## Training
To train the model from scratch, we provide the following script:
```bash
$ chmod +x train.sh
$ ./train.sh
```
Remember to change the ``data_root`` to your own path. 

## Reload ckpts & test_only
To reload checkpoints and only run inference, we provide the following script:
```bash
$ chmod +x test.sh
$ ./test.sh
```
We provide two checkpoints in ``ckpt`` folder: ``train_val.ckpt`` is trained on the whole dataset, and ``train.ckpt`` is trained on the training dataset. You can change the ``test_ckpt_path`` in ``test.sh`` to reload the checkpoints. We provide all checkpoints at this [link](https://cloud.tsinghua.edu.cn/f/58094c10af68494eadab/?dl=1). You can use ``wget https://cloud.tsinghua.edu.cn/f/58094c10af68494eadab/?dl=1`` to download the checkpoints.
We also provide two types of evaluation: ``test`` and ``val``. You can change the ``action`` in ``test.sh`` to choose the evaluation type.
Remember to change the ``data_root`` to your own path. 

## Test face recognition
To test face recognition, you can run ``fr_acc.py`` to get the accuracy of face recognition. 
Remember to change the ``DATA_ROOT`` to your own path.

## Acknowledgement
This code used resources from [BO-QSA](https://github.com/YuLiu-LY/BO-QSA). We thank the authors for open-sourcing their awesome projects.
