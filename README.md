#CUDA加速
### 一、环境安装
NVIDIA驱动安装
~~~bash
ubuntu-drivers devices
sudo apt-get install nvidia-driver-460
~~~
离线安装pytorch-1.8.0+cu111 [[离线下载]](https://download.pytorch.org/whl/torch/)
~~~bash
sudo pip3 install torch-1.8.0+cu111-cp38-cp38-linux_x86_64.whl -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
~~~