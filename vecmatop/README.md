## 输出结果
#### 向量加CPU运行时间
~~~bash
10000000
Runtime: 0.03732(s)
~~~
#### 向量加CUDA运行时间
~~~bash
10000000
threadPerBlock: 256 
blockPerGrid: 39063 
Runtime: 1.8e-05(s)
~~~
#### 矩阵乘法对比
~~~bash
主机端 multiplicationMatrixOnHost: (512,512) (512,512) -> (512,512)                        Runtime: 0.539111 (s)
设备端 multiplicateMatrixOnDevice: (512,512) (512,512) -> (512,512) <<<(256,256), (2,2)>>> Runtime: 0.00156662 (s)
设备端 matrixMultiplyShared:       (512,512) (512,512) -> (512,512) <<<(256,256), (2,2)>>> Runtime: 0.00116493 (s)
设备端 cublasSgemm:                (512,512) (512,512) -> (512,512) <<<(256,256), (2,2)>>> Runtime: 0.000429184 (s)
~~~