.. _header-n0:

Categorical
===========

.. _header-n3:

1.类别特征编码
--------------

   -  类别性特征原始输入通常是字符串形式，除了基于决策树模型的少数模型能够直接处理字符串形式的输入，其他模型需要将类别型特征转换为数值型特征

   -  无序类别特征

   -  有序类别特征

.. _header-n12:

1.1 序号编码(Ordinal Encoding)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  序号编码通常用于处理类别间具有大小关系的特征，序号编码会按照大小关系对类别型特征赋予一个数值
   ID

-  APIs:

   -  ``sklearn.preprocessing.LabelEncoder()``

   -  ``sklearn.preprocessing.``

.. code:: python

   from sklearn.preprocessing import LabelEncoder

   le =  LabelEncoder()
   # 类别型目标特征
   classes = [1, 2, 6, 4, 2]
   new_classes = le.fit_transform(classes)
   print(le.classes_)
   print(new_classes)

.. code:: python

   from sklearn.preprocessing import LabelEncoder

   le =  LabelEncoder()
   # 类别型目标特征
   classes2 = ["paris", "paris", "tokyo", "amsterdam"]
   new_classes2 = le.fit_transform(classes2)
   print(le.classes_)
   print(new_classes1)

.. code:: python

   from sklearn.preprocessing import OrdinalEncoder

   enc = OrdinalEncoder()
   # 类别特征
   classes1 = []

.. _header-n29:

1.2 独热编码(One-Hot Encoding)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  One-Hot Encoding 通常用于处理类别间不具有大小关系的特征

-  One-Hot Encoding
   使用一组比特位，每个比特位表示一种可能的类别，如果特征不能同时属于多个类别，那么这组值中就只有一个比特位是“开”的

-  One-Hot Encoding 的问题是它允许有 k 个自由度，二特征本身只需要 k-1
   个自由度

-  One-Hot Encoding
   编码有冗余，这会使得同一个问题有多个有效模型，这种非唯一性有时候比较难以理解

-  One-Hot Encoding
   的优点是每个特征都对应一个类别，而且可以把缺失数据编码为全零向量，模型输出也是目标变量的总体均值

-  对于类别取值较多的特征的情况下使用 One-Hot Encoding 需要注意：

   -  使用稀疏向量节省空间

   -  配合特征选择降低维度

      -  高纬度特征的问题：

         -  高纬度空间下，两点之间的距离很难得到有效的衡量

         -  模型参数的数量增多，模型变得复杂，容易出现过拟合

         -  只有部分维度对预测有帮助

.. code:: python

   from sklearn.preprocessing import OneHotEncoder
   from pandas import get_dummies

   import pandas as pd

   df = pd.DataFrame({
       "City": ["SF", "SF", "SF", "NYC", "NYC", "NYC", "Seattle", "Seattle", "Seattle"],
       "Rent": [3999, 4000, 4001, 3499, 3500, 3501, 2499, 2500, 2501]
   })
   df["Rent"].mean()
   one_hot_df = pd.get_dummies(df, prefix = "city")
   print(one_hot_df)

.. _header-n59:

1.3 二进制编码(Binary Encoding)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  二进制编码主要分为两步，先用序号编码给每个类别赋予一个类别ID，然后将类别ID对应的二进制编码作为结果。

-  二进制编码本质上是利用二进制对ID进行哈希映射，最终得到 0/1
   特征向量，且维数少于 One-Hot Encoding，节省了存储空间

.. code:: python

   test

.. _header-n67:

1.4 虚拟编码
~~~~~~~~~~~~

-  虚拟编码在进行表示时只使用 k-1
   个自由度，除去了额外的自由度，没有被使用的那个特征通过一个全零向量表示，它称为参照类

-  使用虚拟编码的模型结果比使用 One-Hot Encoding 的模型结果更具解释性

-  虚拟编码的缺点是不太容易处理缺失数据，因为全零向量已经映射为参照类了

.. code:: python

   import pandas as pd

   df = pd.DataFrame({
       "City": ["SF", "SF", "SF", "NYC", "NYC", "NYC", "Seattle", "Seattle", "Seattle"],
       "Rent": [3999, 4000, 4001, 3499, 3500, 3501, 2499, 2500, 2501]
   })
   df["Rent"].mean()
   vir_df = pd.get_dummies(df, prefix = "city", drop_first = True)
   print(vir_df)

.. _header-n77:

1.5 效果编码
~~~~~~~~~~~~

-  效果编码与虚拟编码非常相似，区别在于参照类是用全部由 -1
   组成的向量表示的

-  效果编码的优点是全由-1组成的向量是个密集向量，计算和存储的成本都比较高

.. code:: python

   import pandas as pd

   df = pd.DataFrame({
       "City": ["SF", "SF", "SF", "NYC", "NYC", "NYC", "Seattle", "Seattle", "Seattle"],
       "Rent": [3999, 4000, 4001, 3499, 3500, 3501, 2499, 2500, 2501]
   })
   df["Rent"].mean()
   vir_df = pd.get_dummies(df, prefix = "city", drop_first = True)
   effect_df = vir_df[3:5, ["city_SF", "city_Seattle"]] = -1
   print(effect_df)

.. _header-n84:

1.6 特征散列化
~~~~~~~~~~~~~~

-  散列函数是一种确定性函数，他可以将一个可能无界的整数映射到一个有限的整数范围
   :math:`[1, m]`
   中，因为输入域可能大于输出范围，所以可能有多个值被映射为同样的输出，这称为碰撞

-  均匀散列函数可以确保将大致相同数量的数值映射到 m 个分箱中

-  如果模型中涉及特征向量和系数的内积运算，那么就可以使用特征散列化

-  特征散列化的一个缺点是散列后的特征失去了可解释性，只是初始特征的某种聚合

.. code:: python

   from sklearn.feature_extraction import FeatureHasher

   # 单词特征的特征散列化
   def hash_features(word_list, m):
       output = [0] * m
       for word in word_list:
           index = hash_fcn(word) % m
           output[index] += 1
       return output

   # 带符号的特征散列化
   def hash_features(word_list, m):
       output = [0] * m
       for word in word_list:
           index = hash_fcn(word) % m
           sign_bit = sign_hash(word) % 2
           if sign_bit == 0:
               output[index] -= 1
           else:
               output[index] += 1
       return output


   h = FeatureHasher(n_features = m, input_type = "string")
   f = h.trasnform(df["feat"])

.. _header-n96:

1.7 Helmert Contrast
~~~~~~~~~~~~~~~~~~~~

.. _header-n98:

1.8 Sum Contrast
~~~~~~~~~~~~~~~~

.. _header-n100:

1.9 Polynomial Contrast
~~~~~~~~~~~~~~~~~~~~~~~

.. _header-n101:

1.10 Backward Difference Contrast
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _header-n102:

2.分箱计数
----------

.. _header-n104:

3.特征组合
----------

-  为了提高复杂关系的拟合能力，在特征工程中经常把一阶离散特征凉凉组合，构成高阶组合特征

-  并不是所有的特征组合都有意义，可以使用基于决策树的特征组合方法寻找组合特征，决策树中每一条从根节点到叶节点的路径都可以看成是一种特征组合的方式
