原题链接：[Problem - C - Codeforces](https://codeforces.com/contest/1974/problem/C)

题意：连续三个数组成元组，要是满足两个三元组只有一个数（对应位置）不同，认为该数漂亮，求解一共多少对

# 思路

容斥原理，对三元组可以找
xy, xz, yz 对应位置的两元组统计这些元组的数量，具有相同的两元组的三元组就是所求的三元组，元组数量之和就是和xyz成对的三元组数量

实际上只统计两元组的出现次数会对当前找到的xyz多次统计，每次二元组数量的和是当前xyz对应的，满足条件的对数，但是多统计了xyz自己三次，所以每次减去3

# 实现

```C++
#include <bits/stdc++.h>

using namespace std;

typedef long long LL;
const int N = 2e5 + 10;

  
int t;
int a[N];

int main()
{
    cin >> t;
    while(t --)
    {
        int n;
        cin >> n;
        for(int i = 0; i < n; i ++)
            cin >> a[i];

        map<array<int, 2>, LL> xy, xz, yz;
        map<array<int, 3>, LL> xyz;
        LL cnt = 0;

        for(int i = 0; i + 3 <= n; i ++)
        {
            ++ xy[{a[i], a[i + 1]}];
            ++ xz[{a[i], a[i + 2]}];
            ++ yz[{a[i + 1], a[i + 2]}];

			//与当前找到的xyz成对的个数
            cnt += xy[{a[i], a[i + 1]}];
            cnt += xz[{a[i], a[i + 2]}];
            cnt += yz[{a[i + 1], a[i + 2]}];

            ++ xyz[{a[i], a[i + 1], a[i + 2]}];
            //减掉xyz自身
            cnt -= 3ll*(xyz[{a[i], a[i + 1], a[i + 2]}]);
        }
        cout << cnt << endl;
    }
    return 0;
}
```