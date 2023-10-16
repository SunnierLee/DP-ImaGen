# PRIVIMAGE
PRIVIMAGE is a Differetial Privacy (DP) image generation tool, which leverages the DP technique to generate synthetic data to replace the sensitive data, allowing organizations to share and utilize synthetic images without privacy concerns.
# Requirements
PRIVIMAGE is built using PyTorch 2.0.1 and CUDA 11.8. Please use the following command to install the requirements:
```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt 
```
# Repoduction
We provide an example for how to repoduce the results on CIFAR-10 in our paper. Suppose you had 4 GPUs on your device.
## Data preparations
Download and preprocess CIFAR-10 and ImageNet dataset.
```
# download CIFAR-10 and save it as /data_dir/cifar-10-python.tar.gz
# download ImageNet and save it as a folder /data_dir/imagenet
cd PRIVIMAGE+D
# preprocess CIFAR-10
python dataset_tool.py --source /data_dir/cifar-10-python.tar.gz --dest /data_dir/cifar10.zip
python compute_fid_statistics.py --path /data_dir/cifar10.zip --file /data_dir/cifar10.npz
# prepocess ImageNet and save it as a folder /data_dir/imagenet32
sh pd.sh
```
## Train semantic query function
Train a semantic query function on the public dataset ImageNet.
```
cd ..
cd SemanticQuery
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --nnodes=1 train_imagenet_classifier.py
```
After training, the checkpoints will be saved with the according accuracy on the validate set. You can choose the checkpoint with the highest accuracy to query the semantics. Also you can use our trained checkpoint[url]
## Query semantic
```
python query_semantics.py --weight_file weight_path --tar_dataset cifar10 --data_dir /data_dir/ --num_words 5 --sigma1 484 --tar_num_classes 10
```
The query result will be saved as a .pth file into the folder /QueryResults
## Pre-training
```
cd ..
cd Pre-training

```

# Citation
If you find the provided code or checkpoints useful for your research, please consider citing our paper:
```
...
```
