# 2-Preprocessing

1. [Importar librerías y cargar los datos ](#schema1)

<hr>

<a name="schema1"></a>

# 1. Importar librerías y cargar los datos

~~~python
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
df = pd.read_csv("./data/drawndata1.csv")
~~~
<hr>

<a name="schema2"></a>

# 2. Separamos los datos

~~~python
X = df[['x', 'y']].values
y = df['z'] == "a"

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.savefig("./images/X_y.png")
~~~
![img](./images/X_y.png)


Hay una gran diferencia de escalas entre el eje x e Y, por eso usamos la desviación estandar.

<hr>

<a name="schema3"></a>

# 3. StandardScaler
![img](./images/001.png)
![img](./images/002.png)

~~~python
from sklearn.preprocessing import StandardScaler
X_new = StandardScaler().fit_transform(X)
plt.scatter(X_new[:, 0], X_new[:, 1], c=y)
plt.savefig("./images/X_new_y.png")
~~~
![img](./images/X_new_y.png)


<hr>

<a name="schema4"></a>

# 4. QuantileTransformer
Transformación por quantiles
![img](./images/003.png)

~~~python
from sklearn.preprocessing import  QuantileTransformer
X_new = QuantileTransformer(n_quantiles=100).fit_transform(X)
plt.scatter(X_new[:, 0], X_new[:, 1], c=y);
plt.savefig("./images/new_x_q.png")
~~~
![img](./images/new_x_q.png)