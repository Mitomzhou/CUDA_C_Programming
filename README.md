# CUDA加速
### 一、环境安装
#### GCC & G++ 降级
~~~bash
因为Ubuntu20.04自带的gcc版本为9.3，而cuda10.1不支持gcc-9，因此要手动安装gcc-7
sudo apt-get install gcc-7 g++-7
安装完gcc-7，系统中就存在两个版本的gcc，因此要设置默认的gcc，命令如下：
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 1
此命令可以通过update-alternatives设置gcc各版本的优先级，优先级最高的为系统默认版本，可以用下述命令显示其优先级：
sudo update-alternatives --display gcc
设置默认的g++也是如此：
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 9
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 1
显示g++优先级：
sudo update-alternatives --display g++
~~~
#### NVIDIA-11.2驱动安装
~~~bash
ubuntu-drivers devices
sudo apt-get install nvidia-driver-460
~~~
#### CUDA-10.1安装
~~~bash
离线下载cuda_10.1.105_418.39_linux.run
sudo ./cuda_10.1.105_418.39_linux.run
去掉显卡驱动[x]直至安装完成
然后添加 ~/.bashrc 环境变量
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.1/lib64
export PATH=$PATH:/usr/local/cuda-10.1/bin
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda-10.1
再source ~/.bashrc
检验cuda安装是否成功
nvcc -V
~~~
#### cudnn-7.6.5.32配置
~~~bash
下载cudnn-10.1-linux-x64-v7.6.5.32.tgz后解压缩
sudo cp cuda/include/cudnn.h /usr/local/cuda-10.1/targets/x86_64-linux/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-10.1/targets/x86_64-linux/lib
sudo chmod a+x /usr/local/cuda-10.1/targets/x86_64-linux/include/cudnn.h
sudo chmod a+x /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcudnn*
验证安装是否成功
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
cat /usr/local/cuda-10.1/include/cudnn.h | grep CUDNN_MAJOR -A 2
~~~
#### clion中集成nvcc编译器 [参考[ubuntu clion新建cuda工程]](https://blog.csdn.net/c991262331/article/details/109318565)
~~~bash
在file->setting->build->cmake,在cmake->options中写入:
-DCMAKE_CUDA_COMPILER:PATH=/usr/local/cuda/bin/nvcc
~~~
#### TensorRT-6.0安装
~~~bash
下载源码
git clone -b release/6.0 https://github.com/nvidia/TensorRT
cd TensorRT
git submodule update --init --recursive
下载库解压
[TensorRT-6.0.1.5.Ubuntu-18.04.x86_64-gnu.cuda-10.1.cudnn7.6.tar.gz]
https://developer.nvidia.com/nvidia-tensorrt-download
目录结构
/home/mitom/CUDALesson/TensorRT   #源码目录
/home/mitom/CUDALesson/TensorRT-6.0.1.5 # 库目录
添加环境~/.bashrc变量
# TensorRT-6.0
export TRT_SOURCE=/home/mitom/CUDALesson/TensorRT
export TRT_RELEASE=/home/mitom/CUDALesson/TensorRT-6.0.1.5
export PATH=$PATH:$TRT_RELEASE/lib
编译源码中的sample
cd $TRT_SOURCE
mkdir build && cd build
cmake .. -DTRT_BIN_DIR=$TRT_SOURCE/build/out -DCUDA_VERSION=10.1 -DPROTOBUF_VERSION=3.6.1
(编译报错/bin/sh: python: not found,则执行 sudo ln -s /usr/bin/python3 /usr/bin/python)
make -j8
编译好的可执行文件在$TRT_SOURCE/build/out
----------------------------------------------------------
编译库中的sample
cd $TRT_RELEASE/samples
make -j8
编译好的可执行文件在$TRT_RELEASE/bin,数据在$TRT_RELEASE/data
~~~
### 二、CUDA架构模型
![cuda_model](https://github.com/Mitomzhou/CUDA_C_Programming/blob/master/imgs/cuda_model.jpg)
