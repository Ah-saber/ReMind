原题链接：[1388. 游戏 - AcWing题库](https://www.acwing.com/problem/content/1390/)

题意：两个玩家，最优选择，每次选数列的两端，使得数列选完后自己的分值最大，问各自多少分

# 思路

## 博弈论

两个分别取优，且贪心做不了
考虑博弈论的逻辑
- 所有选法考虑
- 考虑当前选法，选完后对手选最优

在所有选法中选择 选好后，对手也选好后  与对手相比，占据优势最大的选法

优势如何定义？
	这是比较数值的大小，那么选好之后比对手大得多，优势就大

有两种选法，左边或右边，选完后对手选最优，假设对手最优的选法已知，那么选左边与选右边后与对手的数值差就是  **左 - 下一步最优        右 - 下一步最优**

两者取最大即可

## 区间DP

怎么算？

状态表示：存下每个区间最优的选法与下一步最优的选法造成的数值差

状态计算：分左端点与右端点

# 实现

```C++
#include <iostream>
#include <algorithm>
#include <cstring>

using namespace std;

const int N = 110;
int w[N];
int f[N][N];
int n;

int main()
{
    cin >> n;
    for(int i = 0; i < n; i ++) cin >> w[i];
    //为了拓扑上可以遍历到计算所需的点，先遍历区间大小
    for(int len = 1; len <= n; len ++)
        for(int i = 0; i + len - 1 < n; i ++)
        {
        //j 为右端点
            int j = i + len - 1;
            f[i][j] = max(w[i] - f[i + 1][j], w[j] - f[i][j - 1]);
        }
        
    int sum = 0, d = f[0][n - 1];
    for(int i = 0; i < n; i ++)
        sum += w[i];
        
	//sum = A + B, d = A - B，计算A B
    cout << (sum + d) / 2 << ' ' << (sum - d) / 2 << endl;
    
    return 0;
}
```