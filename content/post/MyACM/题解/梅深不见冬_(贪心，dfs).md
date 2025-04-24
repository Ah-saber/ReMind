原题链接：[P5521 [yLOI2019] 梅深不见冬 - 洛谷 | 计算机科学教育新生态 (luogu.com.cn)](https://www.luogu.com.cn/problem/P5521)

题意：一棵树，要在每个节点上都放过一次梅花，每个节点满足当其子节点被放满了才可以放梅花，每个节点要放的梅花数量是其入度的权重值，梅花可以回收，每次只能到父节点或者没有到达过的一个子节点放梅花，求**每个节点**放梅花的最小梅花数量

# 思路

首先这是DFS搜索顺序

然而不同的子节点搜索顺序会有不同的结果

要怎么对子节点排序就是最大的问题

贪心就是要写式子推导证明（理解）而不是光发呆

$$
\begin{align}
&首先简化问题，只对两个子节点的情况做考虑，\\
&我们有节点i，j，其中定义填充i节点所需ans_i，j所需ans_j\\
&权重为w_i,w_j \\
&\textbf{要解决的问题是当i先选为最佳时，要满足什么条件}\\
&当先选i时，父节点最大所需（去掉自身权重）为max(ans_i, w_i+ans_j)\\
&先选j时，为max(ans_j, w_j+ans_i)\\
\end{align}
$$

分类讨论：
- 如果$ans_i$大，那么就有$w_j +ans_j < ans_i$，展开w，可以得到$w_j+w_i+re_j < w_i+re_i = w_j+re_j<re_i$ ，此时先选j的情况有$w_j+re_j,\, w_j+w_i+re_i$ 都满足小于$w_i+re_i =ans_i$
- 如果$w_i+ans_j$大，那么就有$re_j+w_j>re_i$，对于先选j，有两种都要大于它，则有$w_j+re_j >= w_i+w_j+re_j$ 且 $w_j+w_i+re_i>=w_i+w_j+re_j$
	得到不可以选第一个，必须满足$w_j+re_j<=w_j+w_i+re_i$
	化简得到$re_j<=re_i+w_i$ 且 $re_i>=re_j$
	发现只要满足$re_i>=re_j$ 就成立，也就是当满足$re_i>=re_j$的时候，该顺序就是最佳顺序

那就是简单的排序和实现

# 代码

```C++
#include <bits/stdc++.h>

using namespace std;

const int N = 1e5 + 5;

vector<vector<int>> son(N);
vector<int> w(N);
vector<int> ans(N);
int n;

bool cmp(int x, int y)
{
    return ans[x] - w[x] > ans[y] - w[y];
}


void dfs(int x)
{
    for(auto t: son[x])
    {
    //一直找到叶子节点，从下往上更新
        dfs(t);
    }
	//排序
    sort(son[x].begin(), son[x].end(), cmp);

    int res = 0;
    for(auto t: son[x])
    { 
	    //当前还有剩，且够用
       if(res > ans[t])
       {
            res -= w[t];
       } 
       else
       {
       //不够用了，就更新ans，更新后res就等于新子节点剩下的点
            ans[x] += ans[t] - res;
            res = ans[t] - w[t];
       }
    }
    // 根节点自身权重
    ans[x] +=  max(0, w[x] - res);
}



int main()
{ 
    cin >> n;
    for(int i = 2; i <= n ; i ++)
    {
        int x;
        cin >> x;
        son[x].push_back(i); 
    }

    for(int i = 1; i <= n; i ++)
    {
        cin >> w[i];
    }

    dfs(1);

    for(int i = 1; i <= n; i ++)
    {
        cout << ans[i] << ' ';
    }
    cout << endl;

    return 0;
}
```


