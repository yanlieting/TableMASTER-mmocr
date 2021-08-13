一，环境准备
1，Git：https://git-scm.com/download/win
git目录添加到path环境

2，Visual Studio Community 2019：https://visualstudio.microsoft.com/zh-hans/
用于cuda编译以及mmcv编译用；
注意：C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\Hostx86\x64放到path里去。（验证方式，就是cmd里可以直接敲cl命令！目录根据vs2019安装位置和版本可能略有不同。）

3，Anaonda：https://docs.conda.io/en/latest/miniconda.html
用于配置python开发环境,

4，CUDA 10.2：https://developer.nvidia.com/cuda-10.2-download-archive
需要一张显卡，因为mmcv对cpu计算支持的越来越不友好。
查看显卡算力地址：https://developer.nvidia.com/zh-cn/cuda-gpus

二，安装包
Windows10 21H1 (Windows is not officially supported)
Visual Studio 2019
Python 3.7(Python 2 is not supported)
PyTorch 1.9.0
CUDA 10.2
mmcv-full 1.3.9
mmdetection-2.15.1
mmocr-main(或0.2.1)

三，安装步骤
1，构建conda虚拟环境
conda create --name mmcv python=3.7  # 3.6, 3.7, 3.8 should work too as tested
conda activate mmcv  # make sure to activate environment before any operation
删除虚拟环境：

conda remove -n mmcv --all
2，安装pytorch环境
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
当然也可以根据环境，安装旧版本：https://pytorch.org/get-started/previous-versions/

3，安装mmcv
pip install mmcv-full
mmcv的代码编译攻略：https://mmcv.readthedocs.io/en/latest/get_started/build.html

4，安装mmdetection 2.11.0
pip install -r requirements/build.txt
python setup.py develop
四，测试目标检测demo（mmdetection）
faster-rcnn的模型下载地址：
https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
然后放到mmdetection的checkpoints目录下面

1，图片测试：
python demo/image_demo.py demo/demo.jpg configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
2，视频测试：
python demo/video_demo.py demo/demo.mp4 configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth --show
或输出为结果mp4

python demo/video_demo.py demo/demo.mp4 configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth --out demo/result.mp4
注意碰到以下警告时：
UserWarning: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure

解决方案：
就是查找一下import matplotlib.pyplot as plt
在前面加入以下代码

import matplotlib
matplotlib.use('TkAgg')
五，安装mmocr（main，或0.2.1）
pip install -r requirements.txt
这里会碰到无法安装lanms-proper的错误，因为这个第三方库包只适配linux版本。
所以，需要根据https://github.com/open-mmlab/mmocr/pull/189/commits/e3863cfc38677b0118f311082043981b329620ac
把DRRG相关的内容都清理掉！
（issue：https://github.com/open-mmlab/mmocr/issues/379 和 https://github.com/open-mmlab/mmocr/issues/230 来看官方都是没找到替代的方案）

python setup.py develop
测试：https://mmocr.readthedocs.io/en/latest/demo.html#example-1-text-detection

python mmocr/utils/ocr.py demo/demo_text_det.jpg --output demo/det_out.jpg --det TextSnake --recog None --export demo/
六，安装tablemaster-mmocr（平安的表格识别算法， ICDAR 2021 Competition on Scientific Literature Parsing 第2名）
pip install -r requirements.txt
去掉lanms-proper
可能还需要安装：

pip install json-line
七，准备训练数据
1，下载地址：https://dax-cdn.cdn.appdomain.cloud/dax-pubtabnet/2.0.0/pubtabnet.tar.gz?_ga=2.40479009.240398199.1628750833-387107798.1628043078
（10多个G，要下载一会儿）
下载完解压到一个工作目录里。

2，前置处理数据
修改./table_recognition/data_preprocess.py里的’raw_img_root’ 设置为上面的pubnet数据目录；
‘save_root’ 设置为输出目录；
还有一个jsonl_path也需要改为pubnet的数据目录。

python ./table_recognition/data_preprocess.py
3，训练文本、线检测模型（原理pubnet：https://arxiv.org/pdf/1806.02559.pdf）
把./table_recognition/expr目录下的sh文件都拷到./table_recognition/下来。

sh ./table_recognition/table_text_line_detection_dist_train.sh
4，训练文本识别模型（原理：https://arxiv.org/abs/1910.02562）

sh ./table_recognition/table_text_line_recognition_dist_train.sh
5，训练表格识别模型（原理：平安的TableMASTER）

sh ./table_recognition/table_recognition_dist_train.sh
