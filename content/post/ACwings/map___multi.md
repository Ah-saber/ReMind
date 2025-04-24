# 基本用法

- 注意是映射
- 每个元素是 [[pair]]
- 红黑树，查找logn

```c++
# include <map>
	  
insert();  //插入pair
erase(); // 输入pair或是迭代器
find();  

count(); //查找某数，由于不重复，所以返回1表示存在，返回0表示不存在

//常用用法
map<string, int> a;
a[string] = int;    //可以像数组一样用，用key来表示是哪个value
//但是要注意，找元素的时间复杂度是 log n

lower_bound()/ upper_bound();
```

# 排序与自定义排序

[C++的map排序_c++ map排序-CSDN博客](https://blog.csdn.net/chengqiuming/article/details/89816566)

