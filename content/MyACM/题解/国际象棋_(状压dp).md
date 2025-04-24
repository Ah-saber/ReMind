原题链接：[3494. 国际象棋 - AcWing题库](https://www.acwing.com/problem/content/description/3497/)

题意：国际象棋的马，给定k个，要求求解在 N * M 的矩阵中共有多少种马与马之间不互相攻击的摆法

# 思路

看数据范围明显是状态压缩

想办法表示状态，给定只有最大 6 行，也就是说要遍历列

状态压缩要分析出状态之间的相互影响并想办法表示，在这道题中这种关系是马与马的相互攻击

用二进制数来表示每列的马的存放状态，假设遍历到第b列，并且存下了k数量的马。那么这些马是否可以摆放，发现这取决于前两列的马的摆放（一列一列看），只要当前的摆放与之前的摆放不矛盾，就可以作为一种方案

那么要遍历所有可能情况，走到第几列，上上列的状态，上一列的状态，这一列的状态，还有摆放了多少马共5层循环，但是由于矛盾关系会减少大量的时间复杂度

# 实现

```C++
#include <iostream>
#include <algorithm>
#include <cstring>

using namespace std;

const int N = 1 << 6, M = 110, K = 22, mod = 1e9 + 7;
int n, m, k;
int f[M][N][N][K];

//得到一个数1的个数(即马的个数)
int get_one(int b)
{
    int res = 0;
    while(b)
    {
        res ++;
        b -= b & -b;
    }
    return res;
}


int main()
{
    cin >> n >> m >> k;
    //全都是0则方案数唯一
    f[0][0][0][0] = 1;
    
    for(int i = 1; i <= m; i ++)
    {
        for(int a = 0; a < 1 << n; a ++)
        {
            for(int b = 0; b < 1 << n; b ++)
            {
                //去掉会相互攻击的摆法
                if(b & (a << 2) || a & (b << 2)) continue;
                //最上层C在最底层遍历，为了确定下层a,b的值
                for(int c = 0; c < 1 << n; c ++)
                {
                    if(b & (c << 1) || c & (b << 1)) continue;
                    if(a & (c << 2) || c & (a << 2)) continue;
                    
                    //得到当前马的数量
                    int t = get_one(b);
                    for(int j = t; j <= k; j ++)
                    {
                        //当前行b马的个数不变，变之前的个数来实现不同个数马的方案的统计
                        f[i][a][b][j] = (f[i][a][b][j] + f[i-1][c][a][j-t]) % mod;
                    }
                }
            }
        }
    }
    
    //不同选层放置的不同方案
    int ans = 0;
    for(int a = 0; a < 1 << n; a ++)
        for(int b = 0; b < 1 << n; b ++)
            ans = (ans + f[m][a][b][k]) % mod;
            
    cout << ans << endl;
}
```

