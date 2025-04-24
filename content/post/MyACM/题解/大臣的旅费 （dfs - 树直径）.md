原题：[1207. 大臣的旅费 - AcWing题库](https://www.acwing.com/problem/content/1209/)

要求在所给的树形图中找到两个点的距离最远

# 补充概念

## 树的直径

从任一点出发，DFS两次，第二次就是直径所在

证明：[树的直径 - OI Wiki (oi-wiki.org)](https://oi-wiki.org/graph/tree-diameter/)

这道题便是树直径的实现而已，考察DFS和树直径的概念

# 实现

```C++
#include <iostream>
#include <algorithm>
#include <cstring>

using namespace std;

const int N = 1e5 + 10;

int n, mid, d_m;
int h[N], ne[2*N], e[2*N], w[2*N], idx;


void add(int a, int b, int k)
{
    e[idx] = b; ne[idx] = h[a]; w[idx] = k; h[a] = idx ++;
}



void dfs(int x, int fa, int d)
{
    for(int i = h[x]; i != -1; i = ne[i])
    {
        int j = e[i];
        if(j == fa)  continue;
        
        if(d_m < d + w[i])
        {
        //要记录下端点位置和最长路径，由于是往后面找端点，所以数值是随着DFS而增加，而不是返回后再相加
            d_m = d + w[i]; //最长路径
            mid = j; // 更新到直径的端点
        }
        
        dfs(j, x, d + w[i]);
    }
}


int main()
{
    memset(h, -1, sizeof h);
    
    int n;
    cin >> n;
    
    for(int i = 0; i < n - 1; i ++)
    {
        int a, b, k;
        cin >> a >> b >> k;
        //双向
        add(a, b, k);
        add(b, a, k);
    }
    
    dfs(1, -1, 0);
    dfs(mid, -1, 0);
    
    long long res = d_m * (d_m + 1ll) / 2 + d_m * 10;
    
    cout << res << endl;
    
    return 0;
}
```

# 认识

傻逼还是傻逼
基础先学牢固吧