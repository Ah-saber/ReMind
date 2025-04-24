原题连接：[4199. 公约数 - AcWing题库](https://www.acwing.com/problem/content/description/4202/)

题意：给定 a,b ，在每次询问的区间中找到 a, b 的最大的公约数，不存在输出 -1

# 思路

预处理

实际上数据范围并没有非常大，在每次询问前预处理 a b 的公约数就可以

怎么找两个数的公约数呢？

## 理论

两个数的约数可以整除其最大公约数

证明：

![[数学#^eff2d8|算术基本定理]]

利用定理可以发现，两数的最大公约数必定可以表示为公共质因数的幂的乘积的形式

而两者约数也可以，即公约数必定可以整除最大公约数

试除法找约数即可

# 实现

```C++
#include <iostream>
#include <algorithm>
#include <cstring>
#include <vector>

using namespace std;


vector<int> g;

int a, b;
int q, l, r;

int gcd(int a, int b)
{
    return b?gcd(b, a % b):a;
}


int main()
{
    cin >> a >> b;
    //预处理a,b公约数
    
    int mg = gcd(a, b);
    
    for(int i = 1; i <= mg / i; i ++)
    {
        if(mg % i == 0)
        {
            g.push_back(i);
            if(i != mg / i) g.push_back(mg / i);
        }
    }
    
    
    //sort(g.begin(), g.end(), greater<int>() );
    sort(g.begin(), g.end());
    
    cin >> q;
    
    while(q --)
    {
        cin >> l >> r;
        
        /*暴力可以写
        int flag = 0;
        for(int i = 0; i < g.size(); i ++)
        {
            if(g[i] <= r && g[i] >= l)
            {
                flag = 1;
                cout << g[i] << endl;
                break;
            }
        }
        */
        //双指针优化
        int L = 0, R = g.size() - 1;
        
        while(L < R)
        {
            int mid = L + R + 1 >> 1;
            if(g[mid] <= r) L = mid;
            else R = mid - 1;
        }
        
        if(g[L] < l) cout << "-1" << endl;
        else cout << g[L] << endl;
        
        //if(!flag) cout << "-1" << endl;
    }
    
    return 0;
}
```

思路活一点，就算不用最大公约数，这道题也就可以在合理时间内试除法找公约数