原题链接：[Problem - B - Codeforces](https://codeforces.com/contest/1973/problem/B)

题意：给定一个数组，找最小的k，使得任意的相邻k个都满足 或 和 相等

# 思路

发现如果 有一个k成立，那么比k大的k_1就成立
因为 
a,a,a,a,a
   a,a,a,a,a
或出来的结果与
a,a,a,a,a,a 或出来的结果一致，因为上述的a或和相等，或和 相 或 的结果与原本的值一样，也就是下面的a的或和结果

所以k满足单调，小于某个值都不成立，大于某个值都成立

**二分**

那二分的判定条件是什么？
如果k成立，那么相邻的k个或和相等，可以二进制拆开数组中每一个数，做一次前缀和，大于0表示该位在或和中存在，等于0表示不存在

对k做遍历，遍历数组中的相邻k个数，遍历这些数的所有位数，查看是否与第一组数一致

# 实现

```C++
#include <bits/stdc++.h>

using namespace std;
//k成立，大于k的一定成立，交错或和
//成立与否通过两两数组之间是否二进制相同来判断
//预处理各个位数的前缀和

const int N = 1e5 + 10;

int s[N][22];
int a[N];
int t;
int n;


bool check(int x)
{
    for(int i = 1; i <= n - x; i ++)
    {
        for(int j = 0; j < 20; j ++)
        {
            //如果这一位是1
            if(s[x][j] - s[0][j])
            {
                if(!(s[i + x][j] - s[i][j]))
                    return false;
            }
            else
            {
                if(!(s[x][j] - s[0][j]))
                    if(s[i + x][j] - s[i][j])
                        return false;
            }
        }
    }

    return true;
}



int main()
{
    cin >> t;
    while(t --)
    {
        //memset(s, 0, sizeof s);
        cin >> n;
        for(int i = 1; i <= n; i ++)
        {
            cin >> a[i];
            //s[i][0] = a[i] | s[i-1][0];

            for(int j = 0; j < 20; j ++)
            {
                if(a[i] & (1 << j)) s[i][j] = 1;
                else s[i][j] = 0;

                //前缀和
                s[i][j] = s[i-1][j] + s[i][j];
            }
        }

        //二分
        int l = 1, r = n;

        while(l < r)
        {
            int mid = l + r >> 1;
            if(check(mid)) r = mid;
            else l = mid + 1;
        }

        cout << l << endl;
    }

}
```