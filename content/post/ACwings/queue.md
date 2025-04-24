- ### 基本用法
	- ```c++
	  [[include]]<queue>
	  
	  push(); // 向队尾插入一个元素
	  front(); // 返回队头元素
	  back();  //返回队尾元素
	  pop(); // 弹出队头
	  
	  //初始化
	  queue<int> q;
	  
	  //清空(重新构造)
	  q = queue<int>();
	  ```