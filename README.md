# PRIVIMAGE
PRIVIMAGE is a Differetial Privacy (DP) image generation tool, which leverages the DP technique to generate synthetic data to replace the sensitive data, allowing organizations to share and utilize synthetic images without privacy concerns.
# Requirements
PRIVIMAGE is built using PyTorch 2.0.1 and CUDA 11.8. Please use the following command to install the requirements:
```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt 
```
# Repoduction
We provide an example for how to repoduce the results on CIFAR-10 in our paper.
## Data preparations
```
# download CIFAR-10 and save it as /data_dir/cifar-10-python.tar.gz
cd PRIVIMAGE+D
python dataset_tool.py --source /data_dir/cifar-10-python.tar.gz --dest /data_dir/cifar10.zip
python compute_fid_statistics.py --path /data_dir/cifar10.zip --file /data_dir/cifar10.npz
```
## Train semantic query function
We train a semantic query function on the public dataset ImageNet.
```
# download ImageNet and save it as /data_dir/ImageNet

```

# Citation
If you find the provided code or checkpoints useful for your research, please consider citing our paper:
```
...
```
