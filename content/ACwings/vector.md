- ### 基本用法
	- ```c++
	  [[include]]<vector>
	  [[include]]<iostream>
	  
	  using namespce std;
	  
	  vector<int> a(n, m);  //n 长度， m 初始化元素大小
	  size();// 返回元素个数，   复杂度 O(1)
	  empty();  //是否为空
	  clear();  // 清空
	  
	  front()  / back();  //最前与最后
	  push_pack(); //加数
	  pop_back();  //弹出最后
	  begin();  /  end();  //就是 0， 和size()，最后一个数的后一个   这个是迭代器
	  
	  
	  //遍历
	  //迭代器遍历
	  for(vector<int>::iterator i = a.begin(); i != a.end(); i ++)  cout << i << ' ';
	  
	  //比较运算
	  //字典序，类似strcmp
	  a(4, 3)  b(3, 4)   b > a; 
	  ```