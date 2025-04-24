原题链接：[218. 扑克牌 - AcWing题库](https://www.acwing.com/problem/content/220/)‘

题意：找扑克牌，四种花色个A B C D张，其中大小王可以填补任何花色来让数学期望最小，求数学期望

# 思路

列状态转移方程

`f[i]` 为当前状态到满足条件的数学期望
有四种花色和大小王，考虑五维以上数组表示

因为大小王的特殊性，且要求找最小，分开表示更合适

所以六维
`f[A][B][C][D][x][y]` 
x 大王，y小王，用0-3表示放在哪个花色里，4表示没抽到

其中`f[a][b][c][d][x][y]` (当前状态到最终状态的数学期望) 可以由 `f[a+1][b][c][d][x][y],f[a][b+1][c][d][x][y] …… min(f[a][b][c][d][i][y], min(f[a][b][c][d][x][i]` (下一个状态到最终状态的数学期望)转移得到

大小王在遍历好花色的下一个状态后再遍历，可以贪心的得到当前这个状态到最终状态的最小期望选择

初始状态的数学期望就是答案

# 实现

有可能完不成，唯一可能就是牌用完了还没找到
可以返回一个无穷大的数

注意一直找，找到找不到才返回无穷大，中间会乘上系数，最后找没找到的判断不能判断等于无穷大，而是答案大于某个值，一般可以是 大于无穷大 除以2


```C++
#include <iostream>
#include <algorithm>
#include <cstring>

using namespace std;


const int N = 15;
const double INF = 1e20;
double f[N][N][N][N][5][5];

int A, B, C, D;

double dfs(int a, int b, int c, int d, int x, int y)
{
    double &v = f[a][b][c][d][x][y];
    if(v >= 0) return v;
        
    int an  = a + (x == 0) + (y == 0);
    int bn  = b + (x == 1) + (y == 1);
    int cn  = c + (x == 2) + (y == 2);
    int dn  = d + (x == 3) + (y == 3);
    
    if(an >= A && bn >= B && cn >= C && dn >= D) return v = 0;
    
    int res = 54 - a - b - c - d - (x != 4) - (y != 4);
    if(res <= 0) return v = INF;
    v = 1;
	//v初始化为1，因为状态转移方程中有 E(p*(x_1 + 1)),各个状态都遍历到，有 sum(p_i * 1) = 1
	
    if(a < 13) v += (13.0 - a) / res * (dfs(a + 1, b, c, d, x, y) );
    if(b < 13) v += (13.0 - b) / res * (dfs(a, b + 1, c, d, x, y));
    if(c < 13) v += (13.0 - c) / res * (dfs(a, b, c + 1, d, x, y));
    if(d < 13) v += (13.0 - d) / res * (dfs(a, b, c, d + 1, x, y));
    //if(r < 2) v += (2.0 - r) / res * (dfs(a, b, c, d, r + 1));
 
    if(x == 4)
    {
        double t = INF;
        for(int i = 0; i < 4; i ++) t = min(t, 1.0 / res * dfs(a, b, c, d, i, y));
        v += t;
        
    }
    if(y == 4)
    {
        double t = INF;
        for(int i = 0; i < 4; i ++) t = min(t, 1.0 / res * dfs(a, b, c, d, x, i));
        v += t;
    }
 
    return v;
}

int main()
{

    cin >> A >> B >> C >> D;
    
    memset(f, -1, sizeof f);
    double ans = dfs(0, 0, 0, 0, 4, 4);
    if(ans > INF / 2) ans = -1;
    
    printf("%.3lf", ans);
    
    return 0 ;
    
}```

