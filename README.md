# Bidirectionally Learning Dense Spatio-temporal Feature Propagation Network for Unsupervised Video Object Segmentation(ACM MM 2022)

## Prerequisites
The training and testing experiments are conducted using PyTorch 1.8.1 with two GeForce RTX 2080Ti GPUs with 11GB Memory.
- Python 3.6
```
conda create -n dbsnet python=3.6
```


Other minor Python modules can be installed by running
```
pip install -r requirements.txt
```
## Train

### Download Datasets
In the paper, we use the following three public available dataset for training. Here are some steps to prepare the data:

- [DAVIS-16](https://davischallenge.org/davis2017/code.html): We use all the data in the train subset of DAVIS-16. However, please download DAVIS-17 dataset, it will automatically choose the subset of DAVIS-16 for training.
- [YouTubeVOS-2018](https://youtube-vos.org/dataset/): We sample the training data every 5 frames in YoutubeVOS-2018. You can sample any number of frames to train the model by modifying parameter ```--num_frames```.
- [FBMS](https://lmb.informatik.uni-freiburg.de/resources/datasets/moseg.en.html): We use all the data in the train subset of FBMS.

### Prepare Optical Flow
Please following the the instruction of [RAFT](https://github.com/princeton-vl/RAFT) to prepare the optical flow. 

### Prepare pretrained backbond
The pre-trained backbone can be downloaded from [MobileViT backbone](https://github.com/wilile26811249/MobileViT) and put it into the ```pretrained``` folder.

### Train
- First, train the model using the YouTubeVOS-2018, DAVIS-16 and FBMS datasets.
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py
```
- Second, finetune the model using the DAVIS-16 and FBMS datasets.
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py --finetune first_stage_weight_path
```

## Test

-   Run following to generate the segmentation results.
```
python tool.py --checkpoint model_weight_path --tools test
```
- About the post-processing technique DenseCRF we used in the original paper, you can find it here: [DSS-CRF](https://github.com/Andrew-Qibin/dss_crf).

## Segmentation Results

- The segmentation results on DAVIS-16, FBMS, DAVSOD and MCL can be downloaded from [Baidu Pan](https://pan.baidu.com/s/1goQUA1vs6Wg42cSOS0tL4w)(PSW:uf21).
- Evaluation Toolbox: We use the standard UVOS evaluation toolbox from [DAVIS-16](https://github.com/davisvideochallenge/davis-matlab/tree/davis-2016) and VSOD evaluation toolbox from [DAVSOD benchmark](https://github.com/DengPingFan/DAVSOD).
- Note: When we evaluate J_Mean and F_Mean, we need to set [predict_mask > 127] = 255 and [predict_mask <=127] = 0. 

## Citation
If you find DBSNet useful for your research, please consider citing the following papers:
```
@inproceedings{fan2022dbsnet,
  title={Bidirectionally Learning Dense Spatio-temporal Feature Propagation Network for Unsupervised Video Object Segmentation},
  author={Jiaqing Fan, Tiankang Su, Kaihua Zhang, and Qingshan Liu},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={0--0},
  year={2022}
}
```

## Acknowledgments
- Thanks for [DenseNet](https://arxiv.org/pdf/1608.06993.pdf) and [ConvLSTM](https://arxiv.org/pdf/1506.04214.pdf), which helps us to quickly implement our ideas.
