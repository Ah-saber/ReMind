- ### 基本用法
	- ```c++
	  [[include]]<vector>
	  
	  using namespace std;
	  
	  pair<int, string> p;  //随意存放两种元素
	  //初始化
	  p = make_pair(10. "阿尔托莉雅天下第一");
	  p = {10, "阿尔托莉雅天下第一"};
	  
	  //pair 的比较，先first（第一关键字）  second（第二关键字）  按字典序排序
	  //当一个对象有两个属性，又要按某种规则排序，就可以用pair
	  
	  pair<int, pair<int, string>>;  //这样就可以存三个及以上元素
	  ```