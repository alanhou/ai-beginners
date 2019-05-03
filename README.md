# Python之 AI人工智能初学者指南


2019年度要翻译的第五本书已经确定，严格地说这不是一本书，而是一个合辑，在 Packt 上称之为学习路径。书的原名为：

Python: Beginner's Guide to Artificial Intelligence

Build applications to intelligently interact with the world around you using Python

Alan 一直都有深入学习 AI、深度学习、机器学习的想法，但想着那堪忧的高等数学和线性代数相关知识，都没有下定决心。这套书在 Packt 上还一个进阶的高级版，我决定先学习看看难度如何再看是否要去啃那块骨头。我自己把这本书看作是一个挑战，这是迄今我心里最没有谱的一本书，简单地看了一下，这本书是绝不可能不查字典就翻译的，所以在翻译时我可能会穿插在其它书的中间去进行，而不会像前几本书那样翻完一本再翻另一本。

[![Python之 AI人工智能初学者指南](http://upload-images.jianshu.io/upload_images/14565748-4a45d1df01f25bd9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)](http://alanhou.org/homepage/wp-content/uploads/2019/03/2019032415190052.png) 

## 序言

这一学习路径提供了机器学习、深度学习以及现代数据分析所需的实战知识和技术。本书会向读者从头介绍机器学习和深度学习算法，并使用真实和有趣的案例展示如何将它们应用到实际的业务挑战中。读者将找到一个机器学习经典思想和当代洞见的一个新平衡点。我们还将学习通过TensorFlow结合其它的开源Python库构建强大、健壮、精确预测模型。

通过整个学习路径，我们将学习如何使用前馈神经网络（Feedforward Neural Network）、卷积（Convolutional）神经网络, 递归（Recurrent）神经网络和自编码器（Autoencoder）来开发机器学习系统的深度学习应用。并发现如何达到以分布式的方式在GPU上的深度学习编程。

学完这一学习路径，读者将理解AI的基本知识并通过一系列案例研究来帮助读者在真实项目中应用这些技能。

本学习路径包含Packt的如下图书：

*   《通过案例学习人工智能》 作者：Denis Rothman
*   《Python深度学习项目》 作者：Matthew Lamons, Rahul Kumar和Abhishek Nagaraja
*   《TensorFlow人工智能实操》 作者：Amir Ziai, Ankit Dixit

## 本书适用对象

这一学习路径面向所有想要了解人工智能基础知识并通过设计智能方案来实际实施的读者。读者将学习到通过创建实际的AI智能解决方案来扩展机器学习和深度学习知识。发挥这个学习路径的最大作用要求读者有Python和数据统计的相关知识和开发经验。

## 本书主要内容

[第一章 成为一个与时俱进的思想者](https://alanhou.org/adaptive-thinker/)：通过基于马尔科夫决策过程（Markov Decision Process (MDP)）的贝尔曼方程（Bellman equation）来讲解强化学习（reinforcement learning）。案例研究描述如何解决人工驾驶和自动驾驶车辆的快递路线问题。

[第二章 像机器那样思考](https://alanhou.org/think-machine/)：以麦卡洛克-皮特斯（McCulloch-Pitts）神经元作为开场演示神经网络。案例研究为使用神经网络来构建一个仓库环境下贝尔曼方程使用的报酬矩阵。

[第三章 将机器思维应用到人类问题上](https://alanhou.org/apply-machine-thinking-human-problem/)：展示机器预测能力是如何超过人类决策的。案例研为国际象棋的位置及如何将AI程序的结果应用到决策优先级中。

[第四章 成为一个打破惯例的创新者](https://alanhou.org/unconventional-innovator/)：有关从零构建前馈神经网络（FNN）来解决XOR线性分割问题。商业案例为工厂的分组订单。

[第五章 管理机器学习和深度学习的能力](https://alanhou.org/manage-power-machine-learning-deep-learning/)：使用TensorFlow和TensorBoard来构建前馈神经网络（FNN）并在会议中进行展示。

[第六章 别迷失在技术中 - 聚焦优化解决方案](https://alanhou.org/focus-optimizing-solutions/)：使用Lloyd's算法讲解K均值（Kmeans ）聚类程序以及如何对仓库中的自动向导车辆应用优化。

[第七章 何时及如何使用人工智能]()：展示云平台上的机器学习方案。我们会使用Amazon Web Services（AWS） SageMaker来解决K均值聚类问题。业务案例为一家公司如何分析全世界的电话时长。

[第八章 为一些公司设计的变革以及小型到大型公司的颠覆式创新]()：讲解变革性创新和颠覆式创新的区别。

[第九章 让你的神经元投入使用：详细讲解卷积神经网络(CNN)]()：内核、形状、激活函数、池化（pooling）层、平铺（flattening）层和稠密（dense）层。案例研究说明食品公司中CNN的使用。

[第十章 将仿生学应用到人工智能中]()：讲解在展示人类思维时神经科学模型和深度学习方案的不同。TensorFlow MNIST分类器组件的解释后组件并在TensorBoard中详细展示。我们还会涵盖图像、精确度、交叉熵、权重、直方图和图表。

[第十一章 概念表示学习]()：讲解概念表示学习（Conceptual Representation Learning (CRL)），这是一种通过CNN转化为CRL元模块的解决重新流的创新方式。案例研究为如何使用CRLMM进行迁移学习和域学习，扩展模型至调度和自动驾驶汽车。

[第十二章 使用 AI优化区块链]()：有关挖矿区块链以及区块链如何运行。我们使用朴素贝叶斯通过预测交易量来预期仓库级以优化供应链管理（Supply Chain Management (SCM)）区块链的区块。

[第十三章 认知NLP聊天机器人]()：讲解如何实现具有意图、实体和对话流的IBM Watson聊天机器人。我们添加了脚本来自定义对话，添加了情感分析来让系统具有人情味，并使用概念表示学习元模块（CRLMM）来增加对话。

注：NLP (Natural Language Processing) 自然语言处理

[第十四章 加强聊天机器人的情感智能缺陷]()：讲解如何使用同时使用一系列不同的算法来构建复杂对话将聊天机器人转化为一个能够共情的机器。文中讲解受限玻尔兹曼机（Restricted Boltzmann Machines (RBMs)）、CRLMM, RNN、单词转向量（word2Vec）嵌套和主成分分析法（Principal Component Analysis (PCA)）。文中有Python程序描绘机器与用户之前的共情对话。

[第十五章 构建深度学习环境]()：本章中我们将为项目建立一个通用工作空间，核心技术有Ubuntu, Anaconda, Python, TensorFlow, Keras和Google云平台(GCP)。

[第十六章 使用回归训练神经网络进行预测]()：本章中我们会在TensorFlow中构建两层（最小化深度）神经网络（NN），并在经典MNIST数据集的手写数字上训练它，这是一个餐厅服务员文字记录业务用例。

[第十七章 用生成语言模型的内容创作]()：本章中我们将实现生成模型来使用短记忆网络（LSTM）、变分自编码器（VAE）和生成对抗网络（GAN）来生成内容。我们将有效地实现文本和音乐的模型，可生成歌词、艺术家的音乐和各种创意业务。

[第十八章 使用DeepSpeech2构建语音识别]()：本章中我们构建和训练自动语音识别系统来将语音电话转化为文本，然后用于基于文本聊天机器人的输入。通过连接时序分类（Connectionist Temporal Classification (CTC)）损失函数、批标准化以及循环神经网络（RNN）的SortaGrad处理语音和图谱来构建一个端到端的语音识别系统。

[第十九章 使用ConvNets对手写体数字分类]()：本章通过卷积运算、池化和dropout正则化讲解卷积神经网络（ConvNets）的基础。这些都是在工作中优化模型需要调节的杠杆。对比此前第十六章 使用回归训练神经网络进行预测中的深度学习项目的性能结果来了解部署更复杂和更深度模型的价值。

[第二十章 使用OpenCV和TensorFlow进行目标检测]()：本章中我们将学习使用比此前章节信息上更复杂的数据来掌握目标检测和分类，来产生注目的结果。学习使用深度学习包YOLOv2并体验这一模型架构如何更为深度、更为复杂且生成更好的结果。

[第二十一章 使用FaceNet构建面部识别]()：本章中我们将使用FaceNet来构建模型来查看图片并识别其中的面孔，然后执行面部提取来了解图片面部部分的质量。对图片面部识别部分进行特征提取提供与其它数据点（人脸打标签了的图像）比对的基础。这一Python深度学习项目展示了这一技术在从社交媒体到安全应用方面的潜能。

[第二十二章 生成对抗网络]()：讨论扩展卷积神经网络（CNN）的功能。会通过使用CNN来创建合成图像实现。我们将学习如何使用简单的CNN来生成图像。还将来看各种类型的GAN以及它们的应用。

[第二十三章 从GPU到量子计算 - AI硬件]()：讨论可用于AI应用开发的不同硬件。我们将先使用CPU，并扩展至GPU， ASIC和TPU。我们会来看硬件技术相对软件技术的发展。

[第二十四章 TensorFlow Serving]()：讲解如何在服务器上部署训练的模型，这样大多数人可以使用我们的解决方案。我们将学习TensorFlow Serving以及在本地服务器上部署一个非常简单的模型。

