#Siwei_Wang  #MVC 
_Mark  -AS     21:49  11.18  2024_
# Object Function
## Method

![{DBC3A86A-AF36-4644-B9BA-057C5B4A8290}](https://raw.githubusercontent.com/Ah-saber/MyPic/main/%7BDBC3A86A-AF36-4644-B9BA-057C5B4A8290%7D.png)


基于**锚图**的

实现一个 **锚图加权**，**不同锚图大小融合** 的模型

## Focus

- 锚图固定导致学习低下
- 不同大小锚图融合

# Innovation

设计一种方法将不同大小的锚图重构成一个共识锚图，利用共识锚图来做最后聚类

其实类似于投影，将不同维度的锚图投影到一个确定大小来学习共识图

# Idea

- 锚点融合框架，或许有其他融合方法，或许可以有其他融合方法（子空间学习？锚图之间关系学习？）
- 引入R的优化过程或许可以参考


和子空间学习比较像？
锚点融合？