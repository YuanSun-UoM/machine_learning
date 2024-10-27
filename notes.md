# Machine Learning (ML) 机器学习

- Pattern classification (derived from object classification and localization)
- Short-term forecasting (derived from video prediction)

| Supervised                                   | Unsupervised           | Reinforcement                                                                      |
| -------------------------------------------- | ---------------------- | ---------------------------------------------------------------------------------- |
| task driven                                  | data driven            | policy decision                                                                    |
| regression 回归问题, classification 分类问题 | clustering 聚类问题    | maximum reward                                                                     |
| **寻找输入到输出之间的映射**                 | 寻找数据之间的隐藏关系 | **与环境的交互中学习和寻找最佳决策方案**                                           |
| 认知问题                                     |                        | 决策认同                                                                           |
| 已经准备好训练数据输出值                     |                        | 没有准备好训练数据输出值，只有奖励值，并且奖励值不是事先给出的，而是延后计算出的； |
| 训练数据之间相互独立                         |                        | 每一步与时间顺序前后关系紧密                                                       |



# 贝叶斯定理 

## Gaussioan process

- data is not sufficient. we need knowledge. quatifying knowledge

- 先验、后验、似然

- 贝叶斯建模的主要思想是使用一些观察到的数据 D 来推断模型参数θ的后验分布，a probability framework for fitting a model to a training dataset

- 先验

  - 深度高斯过程
  
  - 高斯过程（Gaussian process, GPs）
  
    - GP将一组有限的参数theta从一个连续空间拓展到一个连续无限空间的一个无限函数f，是监督学习方法；
  
    - GPs定义了**先验函数**。观察到某些函数值后，可通过代数运算以将其转换为**后验函数**
  
    - GPs可以用于回归（GP回归，用于连续函数值的推断），也可以用于分类；
  
    - 高斯过程是一种随机过程，其为任意点 分配一个随机变量 ，并且有限数量的这些变量 的联合分布就是高斯分布（即正态分布）
  
      
  
  - multivariate Gaussian distribution
  
  - maximum a posteriori (MAP) 最大后验概率：在贝叶斯统计中，最大后验概率是指给定观测数据后，参数的最有可能的值。
  
    - **curve fitting** 曲线拟合。通过实证确定逼近一组数据的曲线或函数。
  
    - prediction rule: x->y
  
    - Measure of success as probability of misclassified points (ture risk)
  
    - empirical risk minimisation: 经验风险最小化：一种机器学习策略，通过最小化训练数据集上的损失函数来选择模型，以达到在实际应用中的最佳性能。
  
    - we cannot parametrise all possible hypotheisis (假说，假设，猜想) ***H***
  
    - Approximation, estimation, optimisation
  
    - **Marginalisation**: 边缘化
  
    - beta distribution
  
    - gaussian sample
  
    - 当解释变量与误差项呈现出不同程度的相关性时，我们就认为出现了内生性
  
    - “good” parameterisation: 
  
      - **flexible**  such that we do not have to make trade-offs when including beliefs. 
      - **Narrow**
  
    - Non-parametrics: 非参数的
  
      - Non-parametric models 非参数模型: 
  
        - 不对模型做一些预先假设的模型称之为非参数模型，它们可以无约束的去学习，得到各种形式的函数。因此它们在拟合数据的同时也可以得到很强的泛化能力(预测潜在未被学习过的数据点的能力)同样的，列举机器学习中常见的非参数模型以及它们的优势。非参数模型不是说没有参数，只是模型的参数是不固定的且事先没有定义或者这些参数对于我们来说是不可见（不可知）的。参数的数量取决于数据集的大小。
        - k-Nearest Neighbors(K近邻， KNN)
        - Decision Trees(决策树)
        - Support Vector Machines（支持向量机)
        - Gaussian processes: 直接推断函数的分布
        - Gaussian preocess classifier
        - Dirichlet process mixtures
        - inifinite HMMs
        - Infiniter latent factor models
        - 
  
      - parametric model: 
  
        - 预先对模型进行假设的机器学习算法称作parametric machine learning algorithms（参数模型）。若要使用参数模型进行预测，则在用参数模型前就已经知道模型有哪些参数。它们通常包含两个步骤
          1. 预先对模型进行假设
          2. 通过训练数据得到相关系数
  
        - Logistic Regression (逻辑回归)
        - Linear Discriminant (线性判别分析)
        - Perceptron (感知机)
        - Naive Bayes (朴素贝叶斯)
        - Simple Neural Networks (简单神经网络)
        - Polynomial regression: 多项式回归：一种回归分析方法，通过拟合多项式函数来建立自变量和因变量之间的关系。
        - mixture models/k-means
        - hidden markov models: 隐马尔可夫模型
        - factor analysis/PCA/PMF
  
      - Semi-parametric model 半参数模型

- 后验
  - conditional posterior 条件后验
  - predictive posterior 后验预测
- 建立贝叶斯模型的迭代过程
  - 设计你的模型/定义核函数
    - 例如： 平方指数内核**squared exponential kernel**（**高斯内核**或**RBF内核**）
  - 选择先验
    - statistical learning: remove bias
  - 对后验分布进行采样
    - GP 直接生成后验预测分布
  - 检查模型收敛
    - traceplots
    - Rhats
  - 使用后验预测批判性地评估模型并检查它们与您的数据的比较情况
- 模拟数据也是很好的做法，以确保你的模型正确，作为测试你的模型的另一种方式。
  - GCM预报通常都存在误差和不可靠的不确定性传播，因而在使用之前进行校正是非常有必要的。
  - https://mp.weixin.qq.com/s/_juyg-UdlXUqDBXEYBi8Lg


# Supervised ML



# unsupervised ML

### genetic algorithm (GA) 遗传算法

- 进化算法

# Reinforcement Learning 强化学习

- 基于环境而行动，以取得最大化的预期利益；
- 核心思想是智能体`agent`在环境`environment`中学习，根据环境的状态`state`（或观测到的`observation`），执行动作`action`，并根据环境的反馈 `reward`（奖励）来指导更好的动作(如何做出最优的行动选择以获得最大的累积奖励）。
- 从环境中获取的状态（环境可以想象为迷宫）

  - 全局状态state
  - 局部观测值Observation

- 多智能体
- 强化学习没有监督学习已经准备好的训练数据输出值，强化学习只有奖励值；但是这个奖励值和监督学习的输出值不一样，它不是事先给出的，而是延后计算出的。
- Value function





## 算法

- Q-learning 算法



## Related Concept

### policy

- 机器学习中的policy和现实世界中的policy不是相同的概念；
- 

### muti-objective optimization (MOO) 多目标优化

- 寻找最优解
- pareto improvement: 至少有一个目标函数是更优的，同时其他的目标函数不会更差（帕累托改进：在经济学中，指一种资源重新分配的方式，使得至少一个人的状况得到改善，而没有任何人的状况变得更差）
- pareto optimal: 一些列可行解，没有pareto improvement解了
- pareto-optimal front: 非支配解集non-dominated所在的线，这条线内部的一定是dominated solutions

### Value function 价值函数



## AutoML

### FLAML Library

- [https://github.com/microsoft/FLAMLhttps://github.com/microsoft/FLAML](https://github.com/microsoft/FLAML)

In the FLAML automl run configuration, users can specify the task type (任务类型), time budget (时间预算), error metric (误差度量), learner list (学习器列表), whether to subsample (是否进行子采样), resampling strategy type (重新采样策略类型), and so on. All these arguments have default values which will be used if users do not provide them. For example, the default classifiers (分类器) are ['lgbm', 'xgboost', 'xgb_limitdepth', 'catboost', 'rf', 'extra_tree', 'lrl1'].

- time budget: total running time in seconds
- seed: random seed. A seed is a parameter used in algorithms that use random number generators (RNGs随机数生成器) to initialize or control the randomness of the process.
- hyperparmeter: 超参数

### **SHapley Additive exPlanations**（SHAP）

- 机器学习模型解释工具，基于博弈论中的Shapley值，提供了一种分配“公平”的方式，将模型输出的改变分配给每个特征；



- fine tuning
- Physical-informed urban climate emulator
- Physics-guided architecture
- encoder, decoder
- training window, testing window



# Deep learning model

## deep learning framework

- TensorFlow
- PyTorch
- Keras

## tutorial

- [Deep learning in a Nutshell](https://developer.nvidia.com/blog/deep-learning-nutshell-core-concepts/)
- [Deep learning Demystitfied](https://www.nvidia.com/en-us/on-demand/session/gtcfall20-a21323eu/)
  - [slides](./2_tutorials/Deep_Learning_Demystified.pdf)



## UNet (U形卷积神经网络)

- 在气象中常用 https://mp.weixin.qq.com/s/u1y_SwKCHNy1kLvZVS3D7A 



# Foundation model 

- 基础模型：指在机器学习和人工智能领域中，作为基本框架和起点的模型，通常用于进一步优化和改进

Fine-tune 调整；使有规则；对进行微调

- fine tune layers

freezing layers



# Generative AI

- diffussion model 扩散模型：一种用于描述信息、创新或产品在社会中传播和扩散的数学模型。它基于假设，认为个体之间的相互作用和影响是导致信息或创新传播的主要因素。