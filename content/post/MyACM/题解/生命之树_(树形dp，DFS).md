原题链接：[1220. 生命之树 - AcWing题库](https://www.acwing.com/problem/content/description/1222/)

题意：给定一棵树，在连通块中求所有连通块的数值最大的值

# 思路

要从一个点遍历到另一个点的最大数值

## 状态表示

每个点遍历到的连通块的最大值

## 状态计算

dfs 遍历，但是一般的遍历没法实现每个点遍历所有可能通路，同时是无向图，为了防止往回走，使用有 父节点 的dfs

状态计算为当前节点的所有子节点的加和

# 实现

```C++
#include <iostream>
#include <algorithm>
#include <cstring>

using namespace std;
typedef long long LL;

const int N = 1e5 + 10, M = 2 * N;

LL f[N];
int w[N];
int h[N], e[M], ne[M], idx;
int n;

void add(int a, int b)
{
    e[idx] = b; ne[idx] = h[a]; h[a] = idx ++;
}

//dfs遍历所有节点
void dfs(int x, int father)
{
    //每次新遍历到都得是原值
    f[x] = w[x];
    for(int i = h[x]; i != -1; i = ne[i])
    {
        int j = e[i];
        if(j != father)
        {
            dfs(j, x);
        
            f[x] += max(0ll, f[j]);
        }
    }
}

int main()
{
    
    memset(h, -1, sizeof h);
    
    cin >> n;
    for(int i = 1; i <= n; i ++)
        cin >> w[i];
        
    for(int i = 0; i < n - 1; i ++)
    {
        int a, b;
        cin >> a >> b;
    
        //无向图，为了可以从底部往上遍历    
        add(a, b); add(b, a);
    }
    
    
    dfs(1, -1);
    
    LL ans = f[1];
    for(int i = 2; i <= n; i ++)    ans = max(ans, f[i]);
    
    cout << ans << endl;
    
    return 0;
    
}
```