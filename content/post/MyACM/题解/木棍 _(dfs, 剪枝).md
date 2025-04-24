原题链接：[167. 木棒 - AcWing题库](https://www.acwing.com/problem/content/169/)

题意：将小木棍拼成若干根同样长度的木棍，要求长度尽可能小

# 思路

没有什么方法，要使用暴搜+剪枝

dfs暴搜，因为是一次一次选的，用dfs可以加一个循环遍历所有情况
但是只是这么做时间复杂度太高，想办法剪枝

剪枝思路：
- 搜索顺序
	要快就得先决定搜索顺序，可以想到，当先选长的木棍时，后面的木棍的选择就少了，所以从大到小搜
- 判断是否不用再搜（从失败的角度考虑，失败了还要不要再继续找）
	1. 如果当前这一根木棍拼上去后找不到解法，则说明现在的长度是不合理的
	2. 如果当前这根木棍是拼长木棍的第一根，第一根就找不到解法，说明这个长度不合适，直接回溯
	3. 如果当前是最后一根，拼上去后后面的小木棍无解，不管怎么拼这根木棍总得用上，说明这个长度的小木棍总得和别人拼在一起，但是一旦用上了，其它的就无解，所以当前长度也不合理
	4. 长度一致的，判断不合理的小木棍跳过搜索，因为该木棍在当前的拼凑里已经用不了了，长度一样的自然也没法用
	5. 同样的木棍组合有不同顺序，dfs包含顺序，为了减少搜素数，为木棍编号（数组）

要将信息传递给下一个dfs，得知道当前的短木棍拼完了没，当前的长木棍拼到哪个长度，和当前用到哪一根短木棍

# 实现

```C++
#include <iostream>
#include <algorithm>
#include <cstring>

using namespace std;


const int N = 70;

int a[N];
bool st[N];
int sum = 0, len = 0;
int n;
//分别是，当前拼了几根，当前拼到多长，当前是第几根
bool dfs(int u, int cur, int start)
{
    //如果拼完了
    if(u * len == sum)  return true;
    //如果当前木棍拼完了
    if(cur == len) return dfs(u + 1, 0, 0);
    
    for(int i = start; i < n; i ++)
    {
        if(st[i]) continue;
        //如果当前木棍可以选的话
        if(cur + a[i] <= len)
        {
            st[i] = true;
            //如果其后有解
            if(dfs(u, cur + a[i], i + 1)) return true;
            //重置当前的状态
            st[i] = false;
        }
        
        //如果当前i作为第一根或最后一根，选完后后面没解,
        if(!cur || cur + a[i] == len)  return false;
        
        //当前i没解，去掉重复元素
        int j = i;
        while(j < n && a[i] == a[j]) j ++;
        i = j - 1; //因为循环后还有加一
    }
    
    return false;
}

int main()
{

    while(cin >> n, n)
    {
        //memset(a, 0, sizeof a);
        memset(st, false, sizeof st);
        
        sum = 0; len = 0;
        for(int i = 0; i < n; i ++)
        {
            cin >> a[i];
            sum += a[i];
            len = max(len, a[i]);
        }
            
        sort(a, a + n, greater<int>());
        
        for(len; len <= sum; len ++)
        {
            //可以整除
            if(sum % len == 0)
            {
                bool flag = dfs(0, 0, 0);
                
                if(flag)  break;
            }
        }
        
        cout << len << endl;
    }
    return 0;
}
```