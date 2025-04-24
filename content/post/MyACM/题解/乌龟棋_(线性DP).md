原题链接：[312. 乌龟棋 - AcWing题库](https://www.acwing.com/problem/content/description/314/)

题意：给定一行中每点的权值，给定可选的4种前进方式，分别是 1，2，3，4，每种方式可选的数量有限，要求在用完前进方式时得到的总分最大

# 思路

只有一行，如果直接搜索的话，要遍历组合数级别的次数，每种前进方法在所有前进方法里找组合数 $C_{120}^{40}$ 再相乘

复杂度太高

考虑到前面的选择实际上收到后面权重的影响，使用DP，只有一行，考虑线性DP

## 状态表示

一般需要记录当前走到哪个点，各种方式分别用了多少张，但是这么做的话空间会超 （用int数量 * int 字节 / 1024 / 1024）

当当前各种前进方法分别用了多少次已知，规模最大的n（当前走到哪里）就已知，可以去掉最大的一维

## 状态计算

分类，当使用了当前这些方法与其对应次数时，最后一个使用的方法是哪一个进行分类
取最大

# 实现

```C++
#include <iostream>
#include <algorithm>
#include <cstring>

using namespace std;

const int N = 360, M = 45;
int n, m;
int f[M][M][M][M];
int a[N];
int cnt[5];


int main()
{
    cin >> n >> m;
    for(int i = 0; i < n; i ++)
        cin >> a[i];
        
    for(int i = 0; i < m; i ++)
    {
        int x;
        cin >> x;
        cnt[x] ++;
    }
    
    f[0][0][0][0] = a[0];
    for(int A = 0; A <= cnt[1]; A ++)
        for(int B = 0; B <= cnt[2]; B ++)
            for(int C = 0; C <= cnt[3]; C ++)
                for(int D = 0; D <= cnt[4]; D ++)
                {
                    int w = a[1 * A + 2 * B + 3 * C + 4 * D];
                    int &v = f[A][B][C][D];
                    if(A) v = max(v, f[A-1][B][C][D] + w);
                    if(B) v = max(v, f[A][B-1][C][D] + w);
                    if(C) v = max(v, f[A][B][C-1][D] + w);
                    if(D) v = max(v, f[A][B][C][D-1] + w);
                }
    
    cout << f[cnt[1]][cnt[2]][cnt[3]][cnt[4]] << endl;
    
    return 0;
}
```


十分简单的题，但是脑瘫之所以不会写在于不会算空间复杂度，见过的模型也少