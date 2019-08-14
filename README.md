

<!-- MarkdownTOC -->

- Feature Engine
	- 1.Feature Preprocessing
		- 1.1 缺失值处理
		- 1.2 异常值处理
	- 2.Feature Transform
		- 2.1 计数特征二值化【尺度】
		- 2.2 计数特征分箱\(区间量化\)【分布】
			- 2.2.1 固定宽度分箱
			- 2.2.2 分位数分箱
		- 2.3 对数变换【分布】
		- 2.4 指数变换【分布】
		- 2.5 特征缩放【尺度】
			- 2.5.1 Min-Max 特征缩放
			- 2.5.2 标准化\(方差缩放\)
			- 2.5.3 L1/L2 归一化
			- 2.5.4 稳健缩放
	- 3.Feature Building
		- 3.1 简单的特征组合
		- 3.2 多项式特征
	- 4.Feature Selection
		- 4.1 过滤
		- 4.2 打包
		- 4.3 嵌入式
	- 5.Imbalanced Sample
		- 5.1 过拟合
		- 5.2 欠拟合
		- 5.3 SMOTE

<!-- /MarkdownTOC -->


# Feature Engine

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

## 1.Feature Preprocessing


### 1.1 缺失值处理


### 1.2 异常值处理


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


* 区间量化可以将连续型数值映射为离散型数值，可以将这种离散型数值看作一种有序的分箱序列，它表示的是对密度的测量；
* 为了对数据进行区间量化，必须确定每个分箱的宽度：
	- 固定宽度分箱
	- 自适应分箱

```python
from sklearn.preprocessing import KBinsDiscretizer
kbins1 = KBinsDiscretizer(n_bins = 5, encode = "onehot", strategy = "quantile")
kbins2 = KBinsDiscretizer(n_bins = 5, encode = "onehot-dense", strategy = "uniform")
kbins3 = KBinsDiscretizer(n_bins = 5, encode = "ordinal", strategy = "kmeans")
transformed_data = kbins1.fit_transform(data)
```

#### 2.2.1 固定宽度分箱

> * 通过固定宽度，每个分箱中会包含一个具体范围内的数值。这些范围可以人工定制，也可以通过自动分段来生成，它们可以是线性的，也可以是指数性的
> 	- 线性
> 		+ 要将计数值映射到分箱，只需要计数值除以分箱的宽度，然后取整数部分
> 	- 指数
> 		+ 当数值横跨多个数量级时，最好按照10的幂(或任何常数的幂)来进行分组.要将计数值映射到分箱，需要取计数值的对数。

```python
np.floor_divide(X, 10)
```

```python
np.floor(np.log10(X))
```



#### 2.2.2 分位数分箱

> 如果计数数值中有比较大的缺口，就会产生很多没有任何数据的空箱子；
> 根据数据的分布特点，利用分布的分位数进行自适应的箱体定位；


```python
feature = feature.quantile([])
```


### 2.3 对数变换【分布】

> * 对数函数可以对大数值的范围进行压缩，对小数值的范围进行扩展；$x$ 越大，$log_{a}(x)$ 增长得越慢.
> * 对于具有重尾分布的正数值的处理，对数变换是一个非常强大的工具.

* $log_{a}x$:
	- 将 $(0, 1)$ 这个小区间中的数映射到 $(-\infity, 0)$
* $log_{10}x$:
	- 将 $[0, 10]$ 这个区间中的数映射到 $[0, 1]$
	- 将 $[10, 100]$ 这个区间中的数映射到 $[1, 2]$


### 2.4 指数变换【分布】

* 指数变换是个变换足，对数变换只是指数变换的一个特例，它们都是方差稳定变换
* 指数变换可以改变变量的分布，使得方差不再依赖于均值
* 常用的指数变换：
	- 平方根($\sqrt$)变换
	- 对数变换
	- Box-Cox 变换
	- 


### 2.5 特征缩放【尺度】

> * 如果模型对于特征的尺度很敏感，就需要进行特征缩放，顾名思义，特征缩放会改变特征的尺度，也称为特征归一化。
> * 每种特征缩放都会产生一种不同的特征值分布。

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

> * 将特征除以一个归一化常数，比如：$l2$, $l1$ 范数，使得特征的范数为为常数；
> * 归一化不会改变特征的分布；


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

### 3.2 多项式特征



## 4.Feature Selection

> 特征选择技术可以精简掉无用的特征，以降低最终模型的复杂性，它的最终目的是得到一个简约模型，在不降低预测准确率或对预测准确率影响不大的情况下提高计算速度；

### 4.1 过滤


### 4.2 打包


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




## 5.Imbalanced Sample

### 5.1 过拟合


### 5.2 欠拟合


### 5.3 SMOTE



