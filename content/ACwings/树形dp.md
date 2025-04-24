规划问题在于怎么选，选不选

树形dp关键在于节点选不选，由此分出两种状态，利用树形结构将要求的值遍历求解，集中在根节点

# 没有上司的舞会

没有直接上司大家都高兴，求怎么安排大家最高兴
## 状态表示

每个点分为两种状态，在当前遍历到点所有子节点最大值的那种选法中，这个点两种状态的最大值

注意到不是选不选这个点的最大值，因为没考虑到其父节点，只有子节点的最大值已知情况的，其两种情况分别的最大值

## 状态计算

dfs遍历树，`f[i][0] 存下没有第i点的情况的当前树的最大值  and   f[i][1]  存下有第i点的当前树的最大值` 
`f[i][0] += max(f[j][0], f[j][1]`  所有子树互不干扰，其所有子节点的值相加就是当前最大值
`f[i][1] += f[j][0]` 当前值确定选，那么直接下属不选

## 实现

```C++
#include <iostream>
#include <algorithm>
#include <cstring>

using namespace std;

const int N = 6010;

int h[N], e[N], ne[N], idx;

int H[N];
//注意没有告知根节点，要加bool数组求
bool father[N];
int f[N][N];

void add(int a, int b)
{
    e[idx] = b; ne[idx] = h[a]; h[a] = idx ++;   
}

void dfs(int u)
{
    //每个人自己参加幸福程度就加1
    f[u][1] = H[u];
    
    for(int i = h[u]; i != -1; i = ne[i])
    {
        int j = e[i];
        dfs(j);
        
        f[u][0] += max(f[j][0], f[j][1]);
        f[u][1] += f[j][0];
    }
}

int main()
{
    int n;
    cin >> n;
    
    for(int i = 1; i <= n; i ++)
        cin >> H[i];
        
    memset(h, -1, sizeof h);
    
    for(int i = 0; i < n - 1; i ++)
    {
        int a, b;
        cin >> a >> b;
        
        father[a] = true;
        
        add(b, a);
    }
    
    int root = 1;
    while(father[root]) root ++;
    
    //从根节点开始遍历
    dfs(root);
    
    cout << max(f[root][0], f[root][1]) << endl;
    
    return 0;
}
```