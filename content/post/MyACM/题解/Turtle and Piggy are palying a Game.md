原题链接：[Problem - A - Codeforces](https://codeforces.com/contest/1981/problem/A)

题意：找一个范围内质因子最多的数，范围  $2l \leq r$

# 思路

正统思路与证明

For a specific $x$, Piggy always chooses $p$ such that $p$ is a prime number, so the score is the number of prime factors of $x$. It is easy to see that the number with at least $t$ prime factors is $2^t$. The largest integer $t$ satisfying $2^t \le r$ is $\left\lfloor\log_2 r\right\rfloor$. Also, because $2l \le r$, then $\log_2 l + 1 \le \log_2 r$, so $\log_2 l < \left\lfloor\log_2 r\right\rfloor \le \log_2 r$, hence $l < 2^{\left\lfloor\log_2 r\right\rfloor} \le r$. So the answer is $\left\lfloor\log_2 r\right\rfloor$. Time complexity: $O(1)$ or $O(\log r)$ per test case.

# 实现

```C++
#include <bits/stdc++.h>

using namespace std;

const int N = 40;
typedef long long LL;

int t;
LL a[N];

int main()
{
    cin >> t;
    for(int i = 0 ; i < 40; i ++)
        a[i] = 1 << i;

    /* while( t --)
    {
        int l, r;
        cin >> l >> r;

        int x = 0, y = 40;
        
        while(x < y)
        {
            int mid = x + y + 1 >> 1;
            if(a[mid] > r) y = mid - 1;
            else x = mid;
        }

        cout << x << endl;
    } */

    while(t --)
    {
        int l, r;
        cin >> l >> r;

        cout << __lg(r) << endl;
    }
} 
```