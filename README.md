# DBSNet

## Prerequisites
The training and testing experiments are conducted using PyTorch 1.0.1 with two GeForce RTX 2080Ti GPUs with 11GB Memory.
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

- [DAVIS-17](https://davischallenge.org/davis2017/code.html): We use all the data in the train subset of DAVIS-16. However, please download DAVIS-17 dataset, it will automatically choose the subset of DAVIS-16 for training.
- [YouTubeVOS-2018](https://youtube-vos.org/dataset/): We sample the training data every 5 frames in YoutubeVOS-2018. You can sample any number of frames to train the model by modifying parameter **--num_frames**.
- [FBMS](https://lmb.informatik.uni-freiburg.de/resources/datasets/moseg.en.html): We use all the data in the train subset of FBMS.

### Prepare Optical Flow
Please following the the instruction of [RAFT](https://github.com/princeton-vl/RAFT) to prepare the optial flow. 

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
