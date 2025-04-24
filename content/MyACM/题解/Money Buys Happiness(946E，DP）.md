原题链接：[Problem - E - Codeforces](https://codeforces.com/contest/1974/problem/E)

题意：给定一个月的收入，本月收入只能用于之后的支出，支出获得回报，每个月的回报不一样，每个月只能支出一次

# 思路

脑瘫如我看不出来是背包问题

每个月只能支出或不支出，支出获得回报，相当于价值，支出相当于石头的体积，要求在给定的月份里获得最大的回报

就是在每个月会增加容量的背包找石头使得价值最大

**01背包**

但是有问题，给定的体积太大（$10^8$) 没法用数组定义出来，直接用01背包也会超时

逆转一下

把体积当成价值，价值当成体积

即把收入支出当成价值，回报当成体积，求在给定回报的最小支出
当然判断还是用钱够不够用来判断

当求出最小支出后，要是回报对应的支出可以在给定月份后获得，就说明可以得到这些回报，找最大即可

# 实现

```C++
#include <bits/stdc++.h>

using namespace std;

const int N = 1e6 + 10, M = 55;

typedef long long LL;

int t;

/* 其实是贪心
int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0); cout.tie(0);

    cin >> t;
    while(t --)
    {
        LL n, x;
        cin >> n >> x;
        LL m1[M], m2[M];
        
        LL a[M][N] = {};
        LL f[M][N] = {};
        for(int i = 1; i <= n; i ++)
        {
            cin >> m1[i] >> m2[i];
        }

        f[0][0] = f[0][1] = f[1][0] = f[1][1] = 0;
        for(int i = 1; i <= n; i ++)
        {
            if(f[i - 1][0] > f[i - 1][1])
            {
                f[i][0] = f[i-1][0];
                a[i][0] = a[i - 1][0] + x;
            }
            else if(f[i - 1][0] < f[i - 1][1])
            {
                f[i][0] = f[i-1][1];
                a[i][0] = a[i - 1][1] + x;
            }
            else
            {
                f[i][0] = f[i-1][1];
                a[i][0] = max(a[i - 1][0], a[i - 1][1]) + x;
            }

            //如果有钱买
            if(a[i-1][1] >= m1[i])
            {
                f[i][1] = f[i-1][1] + m2[i];
                a[i][1] = a[i-1][1] - m1[i] + x;
            }
            if(a[i-1][0] >= m1[i])
            {
                if(f[i][1] < f[i-1][0] + m2[i])
                {
                    f[i][1] = f[i-1][0] + m2[i];
                    a[i][1] = a[i-1][0] - m1[i] + x;
                }
                else if(f[i][1] == f[i-1][0] + m2[i])
                {
                    a[i][1] = max(a[i-1][0], a[i-1][1]) - m1[i] + x;
                }
            }
        }
        cout << max(f[n][0], f[n][1]) << endl;
    }


    return 0;
} */


LL f[N];
int main()
{
    cin >> t;
    while(t --)
    {
        LL n, x;
        cin >> n >> x;

        LL m1[M], m2[M];
        memset(f, 0x3f, sizeof f);

        LL sum = 0;
        for(int i = 1; i <= n; i ++)
        {
            cin >> m1[i] >> m2[i];
            sum += m2[i];
        }

        LL ans = 0;
        f[0] = 0;
        for(int i = 1; i <= n; i ++)
        {
            for(int j = sum; j >= m2[i]; j --)
            {
                if(f[j - m2[i]] + m1[i] <= (i - 1) * x)
                {
                    f[j] = min(f[j], f[j - m2[i]] + m1[i]);
                    ans = max(ans, (LL)j);
                }
            }
        }

        cout << ans << endl;
    }

    return 0;
}
```
