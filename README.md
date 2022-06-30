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

- [DAVIS-17](https://davischallenge.org/davis2017/code.html): we use all the data in the train subset of DAVIS-16. However, please download DAVIS-17 dataset, it will automatically choose the subset of DAVIS-16 for training.
- [YouTubeVOS-2018](https://youtube-vos.org/dataset/): we sample the training data every 5 frames in YoutubeVOS-2018. You can sample any number of frames to train the model by modifying parameter **num_frames**.
- [FBMS](https://lmb.informatik.uni-freiburg.de/resources/datasets/moseg.en.html): we use all the data in the train subset of FBMS.

## Test
