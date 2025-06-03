## 关于Visual saliency transformer的复现情况 

论文链接：https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Visual_Saliency_Transformer_ICCV_2021_paper.pdf

论文源码：https://github.com/nnizhang/VST

此实验为对VST的复现，我们并未改动原文中模型的大结构，只是为了支持代码在本人的windows环境下运行，我们将NCCL后端更换为gloo；
并修改原仓库中一些旧版本写法，以适应本人的Python 3.12.3版本。

遵照原仓库的结构，我们将训练集放在Data文件夹下，随机分割出700训练集和300测试集，其中训练集使用边缘图来辅助训练，
我们使用cv2.Canny操作数据集的Mask来生成，这些脚本都在truth文件夹下，我们在上传仓库的文件中保留了少量的图片，以展示整体的项目结构。

我们编写脚本对生成的显著性检测结果评分，脚本在RGB_VST/preds下。

要真正运行这一复现后的项目，需要到原论文仓库链接中下下载预计训练模型，并放到VST/pretrained_model下，并在Data中补全全部训练集与测试集。

我们使用下面的指令来复现：
cd VST

cd RGB_VST

python train_test_eval.py --Training True --Testing True --Evaluation True --epochs 200 --train_steps 6000

最终复现的结果为
Average maxF: 0.9111
Average MAE: 0.0392
