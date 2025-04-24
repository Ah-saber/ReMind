- ### 基本用法
	- 是二进制数组
	- 实现了压缩，压缩八个字节到一个字节
![](https://upload-images.jianshu.io/upload_images/11345146-341a596c44c02451.png?imageMogr2/auto-orient/strip|imageView2/2/format/webp)
	- 运行快，空间小
```c++
  #include<bitset>
  
  //位运算基本能用
  
  count(); //返回有多少个1
  any(); //判断是否至少有一个1
  none(); //是否全是0
  
  set(); //全设置为 1
  set(k, v); //将第k位设置为v
  reset(); //把所有位变成0
  flip(); //等价于 ~ （取反）
  flip(k); // 把第k位取反
  
  初始化
  bitset<1000> a;
```
