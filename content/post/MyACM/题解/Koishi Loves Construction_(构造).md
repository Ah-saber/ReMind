原题链接：[P3599 Koishi Loves Construction - 洛谷 | 计算机科学教育新生态 (luogu.com.cn)](https://www.luogu.com.cn/problem/P3599)

题意：给定n，求一组排列，满足前n个的前缀和 mod n 的结果不同，或者前n个前缀积 mod n 的结果不同

# 思路

## 前缀和

首先前缀和，蠢货都看不到排列两个字

容易想到，要mod n 的结果不同，最好就是每两项结果互补，也就是让结果与 n 值 +1， -1，+2，-2……

这样就可以得到各不相同的一组排列

再发现如果n是奇数，前缀和结果是n的倍数，这样mod n结果为0的情况会出现两次，没法构造满足条件的排列，同时，排列的 n值 只能出现在最前，否则另一个数加n mod n 结果相同

得到前缀和的构造

`第一位： n ，偶数位：1，3，5…… idx-1，奇数位：n-1-1, n-2-2…… n-idx+1`

## 前缀积

首先结果不能是n的倍数，否则mod n 结果为0出现两次，最后一项肯定是n，否则n值出现之后，其后的前缀积结果都是0

很难发现其实要求逆元，
这是mod n 同余 i 的问题，每一项前缀积结果要不相同，可以人为定义每一项的结果从1，2，…… n-1，0

这样一来，每次的新乘数，就可以是 当前前缀积 的mod n 余 i 的逆元，也就是等于 mod n余1的逆元乘以i

可以发现这样可以保证每次结果不同，但是能否保证这是一组排列呢？至少可以mod n来保证数不会超过n

其余暂时证明不了

# 代码

```C++
#include <bits/stdc++.h>

using namespace std;

const int N = 100010;

typedef long long LL;

LL n;

void solve_1()
{
    //判断能否成立
    cin >> n;
    //特判1
    if(n & 1 && n != 1) 
    {
        puts("0");
        return;
    }

    cout << "2 ";
    
    //在0上下浮动 0 1 -2 3 -4 这样就可以实现0 1 n-1 2 n-2……
    for(LL i = 1; i <= n; i ++)
    {
        if(i & 1) cout << n + 1 - i;
        else cout << i - 1;
        if(i < n) cout << ' ';
        else cout << endl;
    }
}


//求逆元前要判断素数
bool np[N];
LL prime[N];

void get_prime()
{    
    LL cnt = 0;
    for(LL i = 2; i <= N; i ++)
    {
        if(!np[i]) prime[cnt ++] = i;
        for(LL j = 0; prime[j] <= N / i; j ++)
        {
            //用最小的质数删去其倍数
            np[prime[j] * i] = true;
            if(i % prime[j] == 0) break;
        }
    }
}


LL qmi(LL a, LL k)
{
    LL ans = 1;
    while(k)
    {
        if(k & 1) ans = ans * a % n;
        k >>= 1;
        a = a * a % n;
    }
    return ans;
}


void solve_2()
{
    cin >> n;

    if(np[n] && n != 1 && n != 4)
    {
        puts("0");
        return;
    }

    cout << "2 ";

    if(n == 1)
    {
        cout << "1" << endl; 
        return;
    }
    if(n == 4)
    {
        cout << "1 3 2 4" << endl;
        return;
    }

    for(LL i = 1, ans = 1, sum = ans; i <= n - 1; i ++)
    {
        //求逆元
        cout << ans << ' ';
        ans = qmi(sum, n - 2) * (i + 1) % n;  //因为是下一次的输出，所以i+1
        sum = ans * sum % n;
    }
    cout << n << endl;
}


int main()
{
    LL x, t;
    cin >> x >> t;

    if(x == 1)
        while(t --)
        {
            solve_1();        
        }
    else
    {
        get_prime();
        while(t --)
        {
            solve_2();
        }
    }
        

    return 0;
}
```

