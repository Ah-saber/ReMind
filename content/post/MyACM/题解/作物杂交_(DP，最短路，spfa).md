原题链接：[3305. 作物杂交 - AcWing题库](https://www.acwing.com/problem/content/3308/)

题意：N种作物，给定杂交关系，杂交之后可以得到N种中一种，给定初始作物与相互之间的关系，每种作物的成熟天数，找到最短的时间培养出目标作物

# 思路

题意较为复杂，先读入所有信息再说

不一定能想到这是最短路问题，与一般的最短路不同，其路径是否开通有条件，有两种作物决定一条路

当前选择培育出条件作物之后的最短路不一定需要，说明之后的选择影响最佳选择，是需要考虑当前状态的所有可能性的问题，考虑DP

SPFA，宽搜优化的Dijkstra算法本质就是DP

隐零点，实际上记录的距离是起始当天的时间距离，实现时没有明确指明


依题：状态表示为`d[j] = max(d[x], d[y]) + max(w[x], w[y])`
x和y是两个杂交品种，j为结果品种

结果品种的更新为两个杂交品种最短的诞生时间里长的那个

可能有多杂交品种诞生同一个结果品种，状态转移为所有的可能性找最小


# 实现

```C++
#include <iostream>
#include <algorithm>
#include <cstring>
#include <queue>

using namespace std;

const int N = 2010, M = 2e5 + 10, INF = 0x3f3f3f3f;

typedef long long LL;

int h[N], e[M], w[M], o[M], ne[M], idx;
queue<int> q;
LL d[N];
bool st[N];
bool state[N];
int n, m, k, t;
int l[N];

void add(int a, int b, int c)
{
    //权重为二者的周期最大，记录下通路条件
    e[idx] = b; ne[idx] = h[a]; 
    if(!a) 
        w[idx] = 0;
    else
    w[idx] = max(l[a], l[b]);
    
    o[idx] = c; h[a] = idx ++;
}

int main()
{
    cin >> n >> m >> k >> t;
    
    for(int i = 1; i <= n; i ++)
    {
        cin >> l[i];
    }
    
    memset(d, 0x3f, sizeof d);
    memset(h, -1, sizeof h);
    //记录开过
    for(int i = 0; i < m; i ++)
    {
        int a;
        cin >> a;
        add(0, 0, a);
        d[a] = 0;
        q.push(a);
        st[a] = true;
        //state[a] = true;
    }
    //初始化
   
    
    for(int i = 0; i < k; i ++)
    {
        int a, b, c;
        cin >> a >> b >> c;
        
        add(a, b, c);
        add(b, a, c);
    }
    
    //d[0] = 0;
    //q.push(0);
    //state[0] = true;
    //st[0] = true;
    
    while(q.size())
    {
        auto t = q.front();
        q.pop();
        st[t] = false;
        
        for(int i = h[t]; i != -1; i = ne[i])
        {
            int j = e[i];
            
            //dp状态转移
            if(d[j] > max(d[t], d[o[i]]) + w[i])
            {
                d[j] = max(d[t], d[o[i]]) + w[i];
                if(!st[j])
                {
                    q.push(j);
                    st[j] = true;
                }
            }
        }
    }
    
    cout << d[t] << endl;
    
    return 0;
}
```
