原题链接：[1217. 垒骰子 - AcWing题库](https://www.acwing.com/problem/content/1219/)

题意：垒骰子，一个一个往上加，给定某些组合认为叠放在一起不稳定，给定骰子数，问一共有多少种方案

# 思路

DP

大概是可以看出DP的思路的
	求方案数，前后动态相互影响

## 状态表示

第n次时，选定这一次朝上的面（1-6）后的方案数

## 状态计算

分类为选定的面，选定好这一次的数字后，方案数为上一次的不同面的方案数加和

但是计算要遍历n，题目给定的 n 为 $10^9$ ，必超时

## 矩阵乘法

将DP思想运用在矩阵乘法，想到只有6种分类，可以用矩阵存下这六种类别，同时存下这次选了不同的数的对应结果
	即行为这一次选的数
	列为前一次不同选法后，这一次选了行代表的数的可能方案数

而DP关系式可以实现为
定义可选与否矩阵
	含义是这一次选了这个数后，下一次可以选的数，可选为1，不可选为0

二者相乘可以得到 $A_{n+1}$ 
为 第 n+1 个骰子选定的不同可能与对应的最大方案数

快速幂加取模

**注意**
- 该题认为侧向的四个面顺序不同为不同方案
- 每次选定的数是朝上的数，因为骰子面与面的对应关系，选定朝上的数就确定了朝下的数，要做一个映射关系

# 实现

```C++
#include <iostream>
#include <algorithm>
#include <cstring>

using namespace std;

typedef long long LL;

const int N = 7, mod = 1e9 + 7;

LL a[N][N];
LL f[N][N];
LL ans;
int n, m;

int r[N] = {0, 4, 5, 6, 1, 2, 3};

void mul(LL a[][7], LL b[][7])
{
    LL c[N][N] = {};
    
    for(int i = 1; i <= 6; i ++)
        for(int j = 1; j <= 6; j ++)
            for(int k = 1; k <= 6; k ++)
                c[i][j] = (c[i][j] + a[i][k] * b[k][j]) % mod;
                
    memcpy(a, c, sizeof c);
}

void qmi(int n)
{
	//第一个骰子的选法
    for(int i = 1, j = 1; i <= 6; i ++, j ++)  
        a[i][j] = 4;
	//第一个已选
    n -= 1;
    while(n)
    {
        if(n & 1)  mul(a, f);
        mul(f, f);
        n >>= 1;
    }
}

int main()
{
    cin >> n >> m;
    
    for(int i = 1; i <= 6; i ++)
        for(int j = 1; j <= 6; j ++)
            f[i][j] = 4;
    
    //f[1][4] = f[4][1] = f[2][5] = f[5][2] = f[3][6] = f[6][3] = 0;
            
    for(int i = 0; i < m; i ++)
    {
        int a, b;
        cin >> a >> b;
        //矩阵选了代表上一个的朝上的数后，下一个朝上的数可以选什么
        //所以是一个映射一个不映射
        f[a][r[b]] = 0;
        f[b][r[a]] = 0;
    }
    
    qmi(n);
    
    //mul(a, f);
    
    for(int i = 1; i <= 6; i ++)
        for(int j = 1; j <= 6; j ++)
        {
            //cout << a[i][j] << endl;
            //加起来就是方案数
            ans = (ans + a[i][j]) % mod;
        }
            
    cout << ans << endl;
    
    return 0;
    
}
```