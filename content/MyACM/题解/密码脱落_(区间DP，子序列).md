原题链接：[1222. 密码脱落 - AcWing题库](https://www.acwing.com/problem/content/1224/)

题意：给定本来回文的字符串，现在有些字母没了，问至少没了多少

# 思路

你怎么想不到呢白痴

## 最长回文子序列

子序列不一定连续

找到现有串中最长的回文子序列长度，给定串长 减去 最长回文子序列 就是没有回文的字母数，也就是最少脱落的字母数

## 区间DP

状态表示：此区间的最长回文子序列

状态计算：以区间端点在不在序列中作为分类依据

- 都在
- 只有一个在
- 都不在

分别得到 `f[i+1][j-1] + 2`  `f[i+1][j]   f[i][j-1]  f[i+1][j-1]`
其中 后三者有重叠，用前三者作做状态转移

# 实现

```C++
#include <iostream>
#include <algorithm>
#include <cstring>
#include <string>

using namespace std;

const int N = 1010;

int f[N][N];
string s;

int main()
{
    cin >> s;

    int n = s.size();

    for(int len = 1; len <= n; len ++)
        for(int i = 0; i + len - 1 < n; i ++)
        {
            int j = i + len - 1;
            if(len == 1) f[i][j] = 1;
            else
            {
                f[i][j] = max(f[i + 1][j], f[i][j - 1]);
                //如果回文
                if(s[i] == s[j])
                {
                    f[i][j] = max(f[i][j], f[i + 1][j - 1] + 2);
                }
            }
        }

    cout << n - f[0][n - 1] << endl;

    return 0;
}
```


与最长公共子序列有相似之处
