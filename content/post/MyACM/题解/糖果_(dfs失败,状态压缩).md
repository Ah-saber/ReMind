# 思路

状态压缩

怎么压缩？
用数组存下一个数字来代表二进制数

## 状态表示

一个数组，表示到当前这个状态的包数最小的选法

## 状态计算

遍历所有状态
- 对每个包单选的状态赋值
- 单选的状态赋值后，要是遍历到的状态还没有找到过，证明还没有包可以构成那个状态，跳过
- 当当前状态的值，比原先找到的值更小，或是找到一个还没有找过的状态，就赋值

# 实现

```C++
#include <iostream>
#include <algorithm>
#include <set>
#include <cstring>

using namespace std;

const int N = 110;

int d[N];
int dp[1 << 20];

int n, M, k;


int size(int d)
{
    int sum = 0;
    while(d)
    {
        if(d & 1)
            sum ++;
        d >>= 1;
    }
    
    return sum;
}

bool cmp1(int a, int b)
{
    return a > b;
}

bool cmp2(int a, int b)
{
    return size(a) > size(b);
}



int main()
{
    cin >> n >> M >> k;
    
    for(int i = 0; i < n; i ++)
    {
        for(int j = 0; j < k; j ++)
        {
            int x;
            cin >> x;
            d[i] = d[i] | (1 << (x-1));
        }
        //cout << size(d[i]) << endl;
        
    }
    
    sort(d, d + n, cmp1);
    
    
    //合并相同和包含
    int t = n;
    for(int i = 0; i < n; i ++)
        if(d[i])
            for(int j = i+1; j < n; j ++)
            {
                if(d[j] && (d[i] | d[j]) == d[i])
                {
                    d[j] = 0;
                    t --;
                    //cout << t << endl;
                }
            }
            
    //for(int i = 0; i < n; i ++)
        //cout << d[i] << endl;
    sort(d, d + n, cmp2);
    n = t;
    
    //状态压缩dp
    memset(dp, -1, sizeof dp);
    //初值
    dp[0] = 0;
    int m = 1 << M;
    //cout << m << endl;
    for(int i = 0; i < n; i ++)
    {
        for(int st = 0; st < m; st ++)
        {
            //如果当前状态是没有过的
            if(dp[st] == -1) continue;
            int nst = st | d[i];
            
            //如果选了一包后，当前值没有找到过，或是之前找到的比这种差
            if(dp[nst] == -1 || dp[nst] > dp[st] + 1)  dp[nst] = dp[st] + 1;
        }
    }
    
    //for(int i = 0; i < m;  i++)
       // cout << dp[i] << endl;
    cout << dp[m-1] << endl;
    
    return 0;
}
```


实现中加了之前错误尝试的遗物，可能大概或许会快一些




```C++
//错误的尝试，超时
#include <iostream>
#include <algorithm>
#include <set>
#include <cstring>

using namespace std;

const int N = 110;

struct Candy{
    set<int> kind;
    bool operator > (const Candy &w) const{
        return kind.size() > w.kind.size();
    }
    bool operator == (const Candy &w) const{
        return kind == w.kind;
    }
}C[N];


bool cmp(Candy &a, Candy &b)
{
    return a > b;
}

bool check(Candy a, Candy b)
{
    for(int i = 0; i < b.kind.size(); i ++)
    {
        //if a 包含 b  去掉b
    }
}

bool st[N];
bool fin[N];

int n, M, k;
int m = 0x3f3f3f3f;


bool dfs(int z, int b, Candy t)
{
    if(b + 1 >= m && t.kind.size() < M) return false;
    if(t.kind.size() == M)
    {
        m = b;
        return true;
    }
    
    for(int i = z; i < n; i ++)
    {
        int j = i + 1;
        while(C[j] == C[i] && j < n) j ++;
        i = j;
        
        if(st[i] || fin[i]) continue;
        
        st[i] = true;
        
        Candy tmp = t;
        tmp.kind.insert(C[i].kind.begin(), C[i].kind.end());
        
        dfs(i, b + 1, tmp);
        
        j = i + 1;
        while(C[j] == C[i] && j < n) j ++;
        i = j - 1;
        
        st[i] = false;
    }
    
    return false;
}


int main()
{
    cin >> n >> M >> k;
    for(int i = 0; i < n; i ++)
    {
        for(int j = 0; j < k; j ++)
        {
            int x;
            cin >> x;
            C[i].kind.insert(x);
        }
    }
    
    sort(C, C + n, cmp);
    
    for(int i = 0; i < n; i ++)
    {
        memset(st, 0, sizeof st);
        st[i] = true;
        dfs(i, 1, C[i]);
        fin[i] = true;
            
    }
    
    if(m != 0x3f3f3f3f) cout << m << endl;
    else cout << "-1" << endl;
}



----------------------------------------------------------
//错误代码二 二进制也是不行，剪枝难剪，傻逼也不会剪
#include <iostream>
#include <algorithm>
#include <set>
#include <cstring>

using namespace std;

const int N = 110;

int d[N];

bool st[N];
bool fin[N];

int n, M, k;
int m = 0x3f3f3f3f;

int size(int d)
{
    int sum = 0;
    while(d)
    {
        if(d & 1)
            sum ++;
        d >>= 1;
    }
    
    return sum;
}

bool cmp1(int a, int b)
{
    return a > b;
}

bool cmp2(int a, int b)
{
    return size(a) > size(b);
}


bool dfs(int z, int b, int dt)
{
    int sum = size(dt);
    if(b + 1 >= m && sum < M) return false;
    if(sum == M)
    {
        m = b;
        return true;
    }
    
    for(int i = z; i < n; i ++)
    {
        int big = 0, idx = 0;
        for(int j = i + 1; j < n; j ++)
            if(size(d[i] | d[j]) > big && !st[j]) idx = j;
            
        i = idx;
        
        st[i] = true;
        
        int tmp = dt | d[i];
        dfs(z, b + 1, tmp);
        
        st[i] = false;
    }
    
    return false;
}


int main()
{
    cin >> n >> M >> k;
    
    for(int i = 0; i < n; i ++)
    {
        for(int j = 0; j < k; j ++)
        {
            int x;
            cin >> x;
            d[i] = d[i] | (1 << x);
        }
        //cout << size(d[i]) << endl;
        
    }
    
    sort(d, d + n, cmp1);
    
    
    //合并相同和包含
    int t = n;
    for(int i = 0; i < n; i ++)
        if(d[i])
            for(int j = i+1; j < n; j ++)
            {
                if(d[j] && (d[i] | d[j]) == d[i])
                {
                    d[j] = 0;
                    t --;
                    //cout << t << endl;
                }
            }
            
    //for(int i = 0; i < n; i ++)
      //  cout << d[i] << endl;
    sort(d, d + n, cmp2);
    n = t;
    
    for(int i = 0; i <= n; i ++)
    {
        //cout << size(d[i]) << endl;
        memset(st, 0, sizeof st);
        st[i] = true;
     
        dfs(i, 1, d[i]);
        fin[i] = true;
    }
    
    if(m != 0x3f3f3f3f) cout << m << endl;
    else cout << "-1" << endl;
}
```