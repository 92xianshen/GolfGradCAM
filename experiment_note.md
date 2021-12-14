GolfGradCAM
===========

GradCAM for golfresnet
----------------------
- 时间：2021 年 8 月 12 日
- 目的：对 golfresnet 使用 GradCAM
- 方案：代码在 `src/GolfGradCAM.py` 下
- 结果：失败
- 原因：检查 Gradient of loss w.r.t. the last conv layer，发现数值极小，所以无法响应
- 改进：计划用小模型训练分类器

ConvNet retrained for golf
--------------------------
- 时间：2021 年 8 月 14 日
- 目的：重新训练网络
- 方案：
  - 重新写 CNN 模型：3 x (Conv + ReLU + MaxPool) + 2 x Dense，在 `src/model/ConvNet.py` 中
  - 训练数据集 `dataset/X_train.npz` 和 `dataset/y_train.npz`
  - 训练代码 `src/train.py`
  - 测试代码 `src/inference.py`
- 结果：测试集准确率 ~ 97 %，可以，模型保存在 `saved_model/` 下

GradCAM for ConvNet
-------------------
- 时间：2021 年 8 月 14 日
- 目的：测试 ConvNet 的 GradCAM 表现
- 方案：
  - Load ConvNet
  - 代码 `src/gradcam.py`
- 结果：有效果，可视化结果在 `src/cam_figure/` 下
- 原因：可能是 Dense 层减少，梯度可以传回去了
- 计划：继续改 `gradcam.py` 来记录结果

ConvNet2: Vanilla convolutional net trained on the sensor-wise normalized dataset
---------------------------------------------------------------------------------
- 时间：2021 年 8 月 23 日
- 目的：在逐传感器归一化数据上训练
- 方案：同上
- 代码：`src2`
- 结果：如预期，测试准确率 ~ 97 %
- 计划：下一步测试 GradCAM 表现

GradCAM for ConvNet2
--------------------
- 时间：2021 年 8 月 23 日
- 目的：测试 ConvNet2 的 GradCAM 表现
- 方案：同 GradCAM for ConvNet
- 结果：已保存

B-spline interpolation for GradCAM
----------------------------------
- 时间：2021 年 8 月 24 日
- 目的：将 1D heatmap 插值使用
- 方案：见 `src2/` 中代码
- 结果：完成 B 样条插值

GradCAM for ConvNet2 (Cont'd)
-----------------------------
- 时间：2021 年 8 月 24 日
- 目的：接收 Normalized golf swing 为输入，heatmap 和图片为输出（代码融合）
- 方案：见 `src2/` 中相关代码
- 结果：

2021.11.18迁移本工作至`Work_in_CUMTB/`
=====================================

ConvNet3: convnet with global pooling average layer
---------------------------------------------------
- 时间：2021 年 11 月 18 日
- 目的：继承 ConvNet2 的设置，增加 GAP 层
- 方案：继承 ConvNet2，增加 GAP 层，去掉 Flatten
- 代码：`src3`
- 结果：GAP 层表现不佳，准确率下降
- 计划：继续后面的实验

GradCAM for ConvNet3
--------------------
- 时间：2021 年 11 月 18 日
- 目的：测试 GolfGradCAM
- 方案：继承 GradCAM for ConvNet2
- 代码：`src3`
- 结果：效果并不好？
- 计划：准备考查感受野

GolfVGG: VGG-like net for golf classification with GAP layer
------------------------------------------------------------
- 时间：2021 年 11 月 18 日
- 目的：复现 GolfVGG
- 方案：复现 GolfVGG
- 代码：`src4`
- 结果：效果也很一般，有些 swing 直接失效
- 计划：修改 src4，去掉 GAP 改回 Flatten

GolfVGG: VGG-like net w/ flatten layer
--------------------------------------
- 时间：2021 年 12 月 13 日
- 目的：复现 GolfVGG (w/ flatten layer)
- 方案：
- 代码：src4
- 结果：可用
- 计划：加入 guided grad-cam

Creating Guided Grad-CAM
------------------------
- 时间：2021 年 12 月 13 日
- 目的：加入 guided backpropagation
- 代码：add-gbp 分支