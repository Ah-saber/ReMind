
Huffman树本身就是一个贪心问题，每次选最小的两个合并再参与选最小
要求是分支的权值相等

# 合并果子

原题链接：[148. 合并果子 - AcWing题库](https://www.acwing.com/problem/content/150/)

## 思路

画图，分析出是Huffman问题

![[Huffman树 2024-03-17 11.58.02.excalidraw|300]]
贪心着做
每次选最小，合并后再选最小，Huffman问题
由于权大小一致，那么重量最小的在最深的位置，需要的成本就最低

## 实现

小根堆

```C++
#include <iostream>
#include <algorithm>
#include <queue>
#include <vector>

using namespace std;

int main()
{
    priority_queue<int, vector<int>, greater<int>> heap;
    
    int n;
    cin >> n;
    
    for(int i = 0; i < n; i ++)
    {
        int x;
        cin >> x;
        heap.push(x);
    }
    
    int ans = 0;
    while(heap.size() > 1)
    {
        int a = heap.top(); 
        heap.pop();
        int b = heap.top();
        heap.pop();
        
        ans += a + b;
        
        heap.push(a + b);
    }
    
    cout << ans << endl;
    
    return 0;
}
```
