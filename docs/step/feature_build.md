
## 1. 交互特征

* 两个特征的乘积可以组成一个简单的交互特征，这样可以捕获特征之间的交互作用
* 交互特征的构造非常简单，但是使用起来代价很高

```python
df = pd.DataFrame(
	{},
	columns = ["x1", "x2", "x3", "x4", "x5"]
)
df["x1_x2"] = df["x1"] * df["x2"]
```

## 2. 多项式特征

* 多项式特征可以生成特征的更高阶特征和交互特征

$$\(X_{1}, X_{2}\) => \(1, X_1, X_2, X_{1}^{2}, X_{2}^{2}, X_{1} X_{2}\)$$

```python
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

X = np.arange(6).reshape(3, 2)
poly = PolynomialFeatures(degree = 2, interaction_only = True)
new_X = poly.fit_transform(X)
```

## 3. 自动特征工程

```python
import featuretools as ft
```

### 3.1 5 Minute Quick Start

```python
data = ft.demo.load_mock_customer()
customer_df = df["customers"]
sessions_df = data["sessions"]
transactions_df = data["transactions"]

entities = {
	"customers": (customers_df, "customer_id"),
	"sessions": (sessions_df, "session_id", "session_start")
}
```