# Intro

对于不可分的非线性数据，可以通过非线性变换，将其转换为线性问题，其中将数据映射到高维数据，在高维空间线性可分是一种做法

![[Pasted image 20240901172204.png|500]]
>如图，二维得用椭圆分割，但三维可以用超平面分割，实现线性分割

# Inner Product

两点的内积是具有一定意义的，通过内积可以计算距离和两向量之间的角度

在原空间线性不可分的数据，在新空间中可以线性分割，但是其计算因为维度变高而复杂许多，当讨论到数据间关系等时计算量大，因此提出核函数

核函数就是在低维空间中找到的，可以用来计算高维空间两点内积的函数，即：

$$
K(x_1, x_2) = <\phi(x_1), \phi(x_2)>
$$

用低维空间的计算来代替高维空间的计算，降低计算复杂度

这样的函数被证明是存在的，且不同的函数可以映射到不同维度的空间

# Acknowledgements

[核函数（kernel function） - Lewen - 博客园 (cnblogs.com)](https://www.cnblogs.com/leween/p/12955240.html)

[核函数(Kernel function)(举例说明，通俗易懂)-CSDN博客](https://blog.csdn.net/mengjizhiyou/article/details/103437423)

[通俗理解核方法(kernel function)-CSDN博客](https://blog.csdn.net/u014662865/article/details/80970470)

