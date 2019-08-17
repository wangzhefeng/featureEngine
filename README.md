

<!-- MarkdownTOC -->

- Feature Engine
    - 1.Feature Preprocessing
        - 1.1 异常值处理
            - 1.1.1 异常值出现的原因
            - 1.1.1 异常值检测方法
            - 1.1.2 特征异常值处理方法
        - 1.2 缺失值处理
            - 1.2.1
            - 1.2.2 缺失的类型
    - 2.Feature Transform
        - 2.1 计数特征二值化【尺度】
        - 2.2 计数特征分箱\(区间量化\)【分布】
            - 2.2.1 固定宽度分箱
            - 2.2.2 分位数分箱
        - 2.3 对数变换【分布】
            - 2.3.1 对数变换
            - 2.3.2 指数变换\(Box-Cox\)
        - 2.5 特征缩放【尺度】
            - 2.5.1 Min-Max 特征缩放
            - 2.5.2 标准化\(方差缩放\)
            - 2.5.3 L1/L2 归一化
            - 2.5.4 稳健缩放
    - 3.Feature Building
        - 3.1 简单的特征组合
        - 3.2 多项式特征\(交互特征\)
    - 4.Feature Selection
        - 4.1 过滤
        - 4.2 打包
        - 4.3 嵌入式
    - 5.自动特征工程
    - 6.Imbalanced Sample
        - 6.1 过拟合
        - 6.2 欠拟合
        - 6.3 SMOTE

<!-- /MarkdownTOC -->


# Feature Engine

> 在机器学习中，所有数据最终都会转化为数值型特征，所有特征工程都会归结为某种数值型特征工程技术

* **数值型特征**
    - 特征合理性检查
        + 量级
        + 正负
    - 特征尺度
        + 尺度：
            * 最大值，最小值
            * 是否横跨多个数量级
        + 如果模型是输入特征的平滑函数，那么模型对输入的的尺度是非常敏感的；
        + 使用欧式距离的方法，比如：k均值聚类、最近邻方法、径向基核函数。通常需要对特征进行**标准化**，以便将输出控制在期望的范围内；
        + 逻辑函数对于输入特征的尺度并不敏感。无论输入如何，这种函数的输出总是一个二值变量；
        + 基于空间分割树的模型对尺度是不敏感的；
    - 特征分布
        + 对数变换
        + Box-Cox变换
    - 特征组合
        + 交互特征
    - 特征选择
        + 
* **类别型特征**
    - 分类任务目标变量
* 文本数据
    - 扁平化
    - 过滤
    - 分块




## 1.Feature Preprocessing





### 1.1 异常值处理

> * 异常值的定义：在一个特征的观测值中，明显不同于其他数据或不合乎常理的观测值

#### 1.1.1 异常值出现的原因

* 人为错误
    - 数据输入、记录导致的错误
    - 
* 自然错误
    - 测量误差，比如仪器出现故障


#### 1.1.1 异常值检测方法

* 简单可视化分析
    - 对特征值进行一个数据可视化，远远偏离大部分样本观测值的样本点认为是异常值
* 3 $\sigma$ 原则
    - 当数据服从正态分布，根据正态分布的定义可知，一个观测值出现在距离平均值 3 $\sigma$ 之外的概率是 $P(|x-\mu| > 3\sigma)<=0.003$，这属于极小概率事件，因此，当观测值距离平均值大于 3 $\sigma$，则认为该观测值是异常值；
* 箱型图分析(数字异常值,Numeric Outlier)
    - 落在 (Q1 - 1.5 * IQR) 和 (Q3 + 1.5 * IQR) 之外的观测值认为是异常值
* Z-score
    - 假设特征服从正态分布，异常值是正态分布尾部的观测值点，因此远离特征的平均值。距离的远近取决于特征归一化之后设定的阈值 $Z_thr$, 对于特征中的观测值 $x_i$，如果 $Z_i = \frac{x_i - \mu}{\sigma} > Z_thr$，则认为 $x_i$ 为异常值，$Z_thr$ 一般设为，2.5，3.0，3.5


#### 1.1.2 特征异常值处理方法

* 直接删除含有缺失值的样本
    - 优点：简单粗暴
    - 缺点：造成样本量(信息)减少
* 将异常值当做缺失值，交给缺失值处理方法来处理
    - 优点：
    - 缺点：
* 用特征的`均值`修正；
    - 优点：
    - 缺点：






### 1.2 缺失值处理


#### 1.2.1

* 当缺失数据比例很小时，可直接对缺失记录进行舍弃或进行手工处理
* 实际数据中，缺失数据往往占有相当的比重，这时如果手工处理，非常低效；如果舍弃缺失记录，则会丢失大量信息，使不完全观测数据与观测数据间产生系统差异，对这样的数据进行分析，可能会得出错误的结论

#### 1.2.2 缺失的类型

* 在对缺失数据进行处理前，了解数据缺失的机制和形式是十分必要的。将数据集中不含缺失值的变量称为**完全变量**，数据集中含有缺失值的变量称为不完全变量。
* 从缺失的分布可以将缺失分为：
    - 完全随机缺失(missing completely at random, MCAR)
        + 数据的缺失完全随机的，不依赖任何不完全变量或完全变量，不影响样本的无偏性。如：家庭地址缺失
    - 随机缺失(missing at random, MAR)
        + 数据的缺失不是完全随机的，即该类数据的缺失依赖于其他完全变量。如：财务数据缺失情况与企业的大小有关
    - 完全非随机缺失(missing not at random, MNAR)
        + 数据的缺失于不完全变量自身的取值有关。如：高收入人群不愿意提供家庭收入

对于完全随机缺失和完全非随机缺失，删除记录是不合适的，
随机缺失可以通过已知变量对缺失值进行估计；
非随机缺失没有很好的处理方法；




## 2.Feature Transform

### 2.1 计数特征二值化【尺度】

* 当数据被大量且快速地生成时很有可能包含一些极端值，这时就应该检查数据的尺度，确定是应该保留数据的原始数值形式，还是应该将他们转换为二值数据，或者进行粗粒度的分箱操作；
* 二值目标变量是一个既简单又稳健的衡量指标；


```python
from sklearn.preprocessing import Binarizer
bined = Binarizer(threshod = 1, copy = True)
transformed_data = bined.fit_transform(data)
```


### 2.2 计数特征分箱(区间量化)【分布】

> * 在线性模型中，同一线性系数应该对所有可能的计数值起作用;
> * 过大的计数值对无监督学习方法也会造成破坏，比如:k-均值聚类等基于欧式距离的方法，它们使用欧式距离作为相似度函数来测量数据点之间的相似度，数据向量某个元素中过大的计数值对相似度的影响会远超其他元素，从而破坏整体的相似度测量;
> * 区间量化可以将连续型数值映射为离散型数值，可以将这种离散型数值看作一种有序的分箱序列，它表示的是对密度的测量；
> * 为了对数据进行区间量化，必须确定每个分箱的宽度：
>    - 固定宽度分箱
>    - 自适应分箱

```python
from sklearn.preprocessing import KBinsDiscretizer
kbins1 = KBinsDiscretizer(n_bins = 5, encode = "onehot", strategy = "quantile")
kbins2 = KBinsDiscretizer(n_bins = 5, encode = "onehot-dense", strategy = "uniform")
kbins3 = KBinsDiscretizer(n_bins = 5, encode = "ordinal", strategy = "kmeans")
transformed_data = kbins1.fit_transform(data)
```

#### 2.2.1 固定宽度分箱

> * 通过固定宽度，每个分箱中会包含一个具体范围内的数值。这些范围可以人工定制，也可以通过自动分段来生成，它们可以是线性的，也可以是指数性的
>   - 线性
>       + 要将计数值映射到分箱，只需要计数值除以分箱的宽度，然后取整数部分
>   - 指数
>       + 当数值横跨多个数量级时，最好按照10的幂(或任何常数的幂)来进行分组.要将计数值映射到分箱，需要取计数值的对数。

APIs:

```python
np.floor_divide(X, 10)
np.floor(np.log10(X))
```

Examples:

```python
import numpy as np

# 固定宽度
small_counts = np.random.randint(0, 100, 20)
new_small_counts = np.floor_divide(small_counts, 10)
print(new_small_counts)


# 指数宽度
large_counts = [296, 8286, 64011, 80, 3, 725, 867, 2215, 7689, 
                11495, 91897, 44, 28, 7917, 926, 122, 22222]
new_large_counts = np.floor(np.log10(large_counts))
print(new_large_counts)
```


#### 2.2.2 分位数分箱

> * 如果计数数值中有比较大的缺口，就会产生很多没有任何数据的空箱子；
> * 可以根据数据的分布特点，利用分布的分位数进行自适应的箱体定位
>     - 分位数是可以将数据划分为相等的若干份的数的值

APIs:

```python
pd.qcut()
```

Examples:

```python
import numpy as np
import pandas as pd
large_counts = pd.Series([296, 8286, 64011, 80, 3, 725, 867, 2215, 7689, 
                          11495, 91897, 44, 28, 7917, 926, 122, 22222])
new_large_counts = pd.qcut(large_counts, 4, labels = False)
```


### 2.3 对数变换【分布】

> * 对数函数可以对大数值的范围进行压缩，对小数值的范围进行扩展
> * 对于具有重尾分布的正数值的处理，对数变换是一个非常强大的工具
>     - 与正态分布相比，重尾分布的概率质量更多地位于尾部
>     - 对数变换压缩了分布高端的尾部，使之成为较短的尾部，并将低端扩展为更长的头部，即：经过对数变换后，直方图在重尾的集中趋势被减弱了，在 $x$ 轴上的分布更均匀了一些

* $log_{a}x$:
    - 将 $(0, 1)$ 这个小区间中的数映射到包含全部负数的区间: $(-\infity, 0)$
* $log_{10}x$:
    - 将 $[0, 10]$ 这个区间中的数映射到 $[0, 1]$
    - 将 $[10, 100]$ 这个区间中的数映射到 $[1, 2]$
    - ...

#### 2.3.1 对数变换

APIs:

```python
np.log1p()
np.log10(x + 1)
```


#### 2.3.2 指数变换(Box-Cox)

* 指数变换是个变换族，对数变换只是指数变换的一个特例，它们都是方差稳定变换
* 指数变换可以改变变量的分布，使得方差不再依赖于均值
* 平方根变换和对数变换都可以简单地推广为 Box-Cox 变换
* 常用的指数变换：
    - Box-Cox 变换
        + $x_transformed = \frac{x^{\lambda} - 1}{\lambda}, \lambda \neq 0$
        + $x_transformed = log1p(x), \lambda = 0$
    - 平方根($\sqrt$)变换
        + $\lambda = 0.5$
    - 对数变换(np.log1p(x), np.log10(x + 1))
        + $\lambda = 0$

```python
from scipy.stats import boxcox
# 对数变换
rc_log = boxcox(df["feature"], lmbda = 0)

# Box-Cox:默认情况下，Scipy 在实现 Box-Cox 变换时会找出使得输出最接近于正态分布的 lambda 参数
rc_boxcox = boxcox(df["feature"])
```

* 对比特征的分布与正态分布
    - 概率图(probplot):用于比较特征的实际分布与理论分布，它本质上是一种表示实测分位数和理论分位数的关系的散点图

```python
from scipy import stats
from scipy.stats import probplot
probplot(df["feature"], dist = stats.norn, plot = ax)
```


### 2.5 特征缩放【尺度】

> * 如果模型对于特征的尺度很敏感，就需要进行特征缩放，顾名思义，特征缩放会改变特征的尺度，也称为特征归一化
> * 不论使用何种特征缩放方法，特征缩放总是将特征除以一个常数(归一化常数)，因此不会改变单特征的分布
> * 与对数变换不同，特征缩放不改变分布的形状，只有数据尺度发生了变化
> * 当一组输入特征的尺度相差很大时，就需要进行特征缩放


#### 2.5.1 Min-Max 特征缩放

> * Min-Max 特征缩放可以将特征值压缩(或扩展)到$[0, 1]$区间中；

$$\frac{x - min(x)}{max(x) - min(x)}$$

```python
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
featureScaled = mms.fit_transform(feature)
```

#### 2.5.2 标准化(方差缩放)

> * 标准化后的特征均值为0，方差为1；
> * 如果初始特征服从正态分布，标准化后的特征也服从正态分布(标准正态分布);
> * 不要中心化稀疏数据;

$$\frac{x - mean(x)}{\sqrt(var(x))}$$

```python
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
featureScaled = ss.fit_transform()(feature)
```


#### 2.5.3 L1/L2 归一化

> * 将特征除以一个归一化常数，比如：$l2$, $l1$ 范数，使得特征的范数为为常数
> * 归一化不会改变特征的分布

```python
from sklearn.preprocessing import Normalize

```


#### 2.5.4 稳健缩放

> * 对于包含异常值的特征，标准化的效果不好，可以使用稳健的特征缩放技术对特征进行缩放；

```python
from sklearn.preprocessing import RobustScaler
rs = RobustScaler(with_centering = True,
                  with_scaling = True,
                  quantile_range = (25.0, 75.0),
                  copy = True)
transform_data = rs.fit_transform(data)
```




## 3.Feature Building


### 3.1 简单的特征组合

### 3.2 多项式特征(交互特征)

* 两个特征的乘积可以组成一个简单的交互特征，这样可以捕获特征之间的交互作用
* 交互特征的构造非常简单，但是使用起来代价很高

```python
from sklearn.preprocessing import PolynomialFeatures
```


## 4.Feature Selection

> 特征选择技术可以精简掉无用的特征，以降低最终模型的复杂性，它的最终目的是得到一个简约模型，在不降低预测准确率或对预测准确率影响不大的情况下提高计算速度；

### 4.1 过滤

* 对特征进行预处理，除去那些不太可能对模型有用处的特征
    - 计算特征与相应变量之间的相关性或互信息，然后过滤掉那些在某个阈值之下的特征
    - 没有考虑模型，可能无法为模型选择出正确的特征


### 4.2 打包

* 试验特征的各个子集


### 4.3 嵌入式

> 将特征选择作为模型训练过程的一部分；

* 特征选择是决策书与生俱来的功能，因为它在每个训练阶段都要选择一个特征来对树进行分割；
    - 决策树
    - GBM
    - XGBoost
    - LightGBM
    - CatBoost
    - RandomForest
* $L1$ 正则化可以添加到任意线性模型的训练目标中，$L1$ 正则化鼓励模型使用更少的特征，所以也称为稀疏性约束；
    - LASSO

## 5.自动特征工程


## 6.Imbalanced Sample

### 6.1 过拟合


### 6.2 欠拟合


### 6.3 SMOTE



