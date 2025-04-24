原题链接：[P4552 [Poetize6] IncDec Sequence - 洛谷 | 计算机科学教育新生态 (luogu.com.cn)](https://www.luogu.com.cn/problem/P4552)

题意：给定一个数列，可以任意选择一个区间，让数字都加一或减一，问操作多少次才能使数列中数字都一样，求在保证次数最少的情况，有多少种可能

# 思路

其实就是要求差分数组中除第一个数之外，其他数都为0

题目中的操作其实就是对差分数组左端 l 和右端点 r 进行操作

第一个数的差分其实就是其本身，要让其他数的差分都为0，可以采取的策略如下：
- 正数与负数的差分数组优先操作
	因为每次这样操作，可以对正数减，对负数加，整个数组都变为0的速率快于对同符号的做加减
- 正数负数一方结束了
	- 余下的数与第一项做运算
	- 余下的数与n+1项做运算（就是区间右端点选为数组右端，这样可以不对右端点做操作）

这样就可以归结为：
- 最少次数为`max(p, ne)`
- 情况为：选多少个对数组第一项做运算，每做一次，结果的数值就不一样，也就是`abs(p - ne)+1` 其中 +1 是因为可以完全不做运算

```C++
//题解
#include <bits/stdc++.h>

using namespace std;

const int N = 1e5 + 10;

typedef long long LL;

LL a[N];
LL c;

int main()
{
    int n;
    cin >> n;

    LL p = 0, ne = 0;

    for(int i = 1; i <= n; i ++)
    {
        cin >> a[i];
        //除了差分c1
        if(i > 1)
        {
            c = a[i] - a[i - 1];
            if(c > 0) p += c;
            else ne -= c;
        }
    }

    LL ans1 = max(p, ne);
    LL ans2 = abs(p - ne) + 1; //对b1操作或完全不操作之间

    cout << ans1 << endl << ans2 << "\n";

    return 0;
}
```
