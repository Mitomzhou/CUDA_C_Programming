#CUDA加速
### 一、环境安装
GCC & G++ 降级
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
NVIDIA-11.2驱动安装
~~~bash
ubuntu-drivers devices
sudo apt-get install nvidia-driver-460
~~~
CUDA-10.1安装
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
cudnn-7.6.5.32配置
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
clion中集成nvcc编译器 [参考[ubuntu clion新建cuda工程]](https://blog.csdn.net/c991262331/article/details/109318565)
~~~bash
在file->setting->build->cmake,在cmake->options中写入:
-DCMAKE_CUDA_COMPILER:PATH=/usr/local/cuda/bin/nvcc
~~~
### 二、CUDA架构模型
![cuda_model](https://github.com/Mitomzhou/CUDA_C_Programming/blob/master/imgs/cuda_model.jpg)
