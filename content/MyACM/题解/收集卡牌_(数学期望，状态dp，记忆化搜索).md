原题链接：[4009. 收集卡牌 - AcWing题库](https://www.acwing.com/problem/content/4012/)

题意：抽卡游戏，一直到收集到全部卡牌，每张卡抽到的概率已知，抽到重复的卡就换算为硬币，每k个硬币可以换一张卡，在现有硬币换出去后可以收集到全部卡牌才换

# 思路

数学期望问题

对数学期望列公式化简，可以得到如下：
![[3875d470bb1760c98f1844895b07dc1.jpg]]

得到dp关系式，即可用DP求解

状态压缩为 01 串，只表示收没收集到
如果访问过，那么就是硬币数加一，剩余卡牌的数量不变，如果没有，那么状态变化，剩余卡牌数量减一

# 实现

```C++
#include <iostream>
#include <algorithm>
#include <cstring>

using namespace std;

const int N = 1 << 16, M = 90;

double f[N][M];
double p[N];

int n, k;

double dp(int state, int coin, int r)
{
    double &v = f[state][coin];
    //已经找过了
    if(v >= 0) return v;
    
    //硬币够了
    if(coin >= r * k) return v = 0;
    
    //因为本来是个负数，所以要赋值
    v = 0;
    for(int i = 0; i < n; i ++)
        if(state >> i & 1)
            v += p[i] * (dp(state, coin + 1, r) + 1);
        else 
            v += p[i] * (dp(state | (1 << i), coin, r - 1) + 1);
            
    return v;
}

int main()
{
    cin >> n >> k;
    
    for(int i = 0; i < n; i ++)
        cin >> p[i];
    
    memset(f, -1, sizeof f);
    printf("%.10lf", dp(0, 0, n));
    
    return 0;
}
```

对题目公式要进一步分析

数学期望问题也是一类问题