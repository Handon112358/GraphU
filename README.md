Virtual Env:
##### 首先创建名为pyg36的python虚拟环境

conda create -n pyg36 python=3.6.10

##### 然后在环境中配置GPU版本的pytorch包依赖

pytorch == 1.9.0+cu111
torch-geometric == 2.0.3
torchvision == 0.10.0+cu111

##### (optional) 直接使用conda源速度较慢，切换pip清华源加速安装过程可设置：

pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

##### 使用清华源安装上述的三个包：（实际安装时' == '会报错，'=='是合法）

pip install torch == 1.9.0+cu111 torchvision == 0.10.0+cu111 torch-geometric == 2.0.3 -f https://download.pytorch.org/whl/torch_stable.html

##### 接下来在环境中配置上述三个包的稀疏张量依赖pytorch包

torch-sparse == 0.6.10
torch-scatter == 2.0.7 
由于版本更新问题，导致这些包在目前的在线源中多半不存在，需要下载后离线安装

##### 具体操作：

进入https://data.pyg.org/whl/index.html网址中，寻找不排斥的安装版本，由于torch的版本为1.9.0，单击并进入torch-1.9.0+cu111：
torch_sparse-0.6.10-cp36-cp36m-linux_x86_64.whl
torch_scatter-2.0.7-cp36-cp36m-linux_x86_64.whl（若系统为linux版本则保持不变，windows版本请选择后缀为-win_amd64.whl）

##### 下载后，在本地安装，安装命令为：

pip install+本地目录，例：

pip install /root/autodl-fs/graphU/GIF/GIF-torch-main/package/torch_scatter-2.0.7-cp36-cp36m-linux_x86_64.whl
pip install /root/autodl-fs/graphU/GIF/GIF-torch-main/package/torch_sparse-0.6.10-cp36-cp36m-linux_x86_64.whl

##### 收尾工作，安装一些基础的包依赖：

pip install ogb

##### 环境到此配置完毕


Baselines 

(**) must compare

(*) could compare

(o) maybe compare

(1) Re-Training.


(2) Exact Unlearning.
1. GraphEraser (SISA) (**)
2. Inductive Graph Unlearning (o)


(3) Certified Unlearning
1. Certified graph unlearning (CGU) Eli Chien (**)
2. Certified edge unlearning for graph neural networks (CEU)
3. GIF (*)
4. IDEA (*)
