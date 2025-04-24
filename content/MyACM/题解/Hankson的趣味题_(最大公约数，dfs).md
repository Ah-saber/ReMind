原题链接：[200. Hankson的趣味题 - AcWing题库](https://www.acwing.com/problem/content/description/202/)

题意：给定 a0, a1, b0, b1 ，找到 x 使得 x与a0最大公约数是 a1，与b0最小公倍数是b1，统计这样的 x 的数量


# 思路

x 与 a0 的公约数 是 a1 ，与b0的公倍数的 b1

考虑直接遍历，b1的约数大概1600个，找约数大概要 根号 的复杂度，相乘之下可能会超时，且有多次询问

优化找约数方法，先求质数，考虑到约数就是质因子的不同幂次乘积，可以先对最大的范围预处理找质数，对每个询问试除法找质因数，再用dfs计算不同的幂次组合相乘的结果（即约数），就可以找到所有约数

时间复杂度：

有质数定理：
> 1~n 中质数个数约为 $n / ln(n)$ 个

可以求出预处理范围，同时时间复杂度为：
- 线性筛求质数 ，时间为质数个数 $n/ln(n)$
- dfs求约数，时间为约数个数 1600
- n组数据 2000

dfs与求质数在时间上平行
即最终时间复杂度约为 $10^7$

# 实现

```C++
#include <iostream>
#include <algorithm>
#include <cstring>

using namespace std;

typedef pair<int, int> PII;
typedef long long LL;


//凭经验给出约数,质数最大个数和质因数最大个数
const int N = 45000, M = 50;

int p[N];
PII factor[M];

int cntf; //记录质因子是
int divider[N], cntd;

int n;
bool st[N];


void get_prime(int n)
{
    int cnt = 0;
    for(int i = 2; i <= n; i ++)
    {
        if(!st[i]) p[cnt ++] = i;
        for(int j = 0; p[j] <= n / i; j ++)
        {
           st[p[j] * i] = true;
           if(i % p[j] == 0) break;
        }
    }
}

//对素数试除法
void get_factor(int n)
{
    cntf = 0;
    for(int i = 0; p[i] <= n / p[i]; i ++)
    {
        int pr = p[i];
        //如果是质因子,求其幂范围
        if(n % pr == 0)
        {
            int s = 0;
            while(n % pr == 0) s ++, n /= pr;
            factor[++ cntf] = {pr, s};
        }
    }
    //如果剩下的这个也是约数
    if(n > 1) factor[++ cntf] = {n, 1};
}


//对素数所有可能幂数组合的取值进行搜索
void dfs(int u, int p)
{
    //如果u已经遍历过了所有质数
    if(u > cntf)
    {
        divider[cntd ++] = p;
        return;
    }
    
    for(int i = 0; i <= factor[u].second; i ++)
    {
        //对每个素数的不同幂次组合
        dfs(u + 1, p);
        p *= factor[u].first; 
    }
}



int gcd(int a, int b)
{
    return b?gcd(b, a % b):a;
}


int main()
{
    cin >> n;
    
    get_prime(N);
    
    while(n --)
    {
        int a0, a1, b0, b1;
        cin >> a0 >> a1 >> b0 >> b1;
        
        get_factor(b1);
        
        cntd = 0;
        dfs(1, 1);
        
        int cnt = 0;
        for(int i = 0; i < cntd; i ++)
        {
            LL x = divider[i];
            if(gcd(x, a0) == a1 && (LL)x * b0 / gcd(x, b0) == b1)
                cnt ++;
        }
        
        cout << cnt << endl;
    }
    
    return 0;
}
```

注意此题提供一种新的约数求法

同时利用查询前预处理可以降低时间复杂度