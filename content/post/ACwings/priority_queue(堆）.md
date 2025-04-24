- ### 基本用法
- 
```c++
  [[include]] <queue>
  
  push();//插入元素
  top(); // 返回堆顶元素
  pop(); // 弹出堆顶元素
  
  //初始化，默认  -大根堆-   从大到小排序
  priority_queue<int> heap;
  //当要小根堆时
  heap.push(-x); //每次插入负数，再从大到小排序就相当于从小到大排序（原来的元素是正数）
  
  //或者
  priority_queue<int, vector<int>, greater<int>> heap; //就直接是小根堆了   ？
  ```