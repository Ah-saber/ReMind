原题链接：[5556. 牛的语言学 - AcWing题库](https://www.acwing.com/file_system/file/content/whole/index/content/11444499/)

题意：给定字符串，前缀大于4，后缀只能为2或3，并且连续两个后缀不重复，找到所有可能后缀并输出

# 思路

想吧，是dp问题

如何得知是dp问题？
分析得到，每个字符串能否是后缀与其后字符串的状态有关，可以归为地推问题，而递推是dp的特例

一个串能否是后缀之一，取决于两点
- 其前面是否能作为前缀
- 其后面是否能作为后缀并且与它连续重复

1. 是否能作为前缀
	能作为前缀，则第一个后缀就是当前字符串，并且长度大于4
2. 是否后缀
	- 要构成后缀，首先得其自身满足串可以有数个后缀组成
	- 并且该串后面的串不可以与其相同：
		- 若是相同，则看取另一个长度，一定不同
		- 而另一长度这种取法是否成立，取决于取另一长度后，其后是否能够成后缀串

由此写出代码
```C++
#include <iostream>
#include <algorithm>
#include <cstring>
#include <set>

using namespace std;

const int N = 1e5 + 10;

bool f[N];


int main()
{
    string s;
    cin >> s;
    int n = s.size();
    //cout << n;
    
    set<string> ans;
    f[n-1] = true;
    
    for(int i = n - 2; i >= 4; i --)
        for(int j = 2; j <= 3; j ++)
        {
            if(f[i + j])
            {
                //判断是否当前串与之后的同长度的串是否相同
                string t1 = s.substr(i + 1, j);
                string t2 = s.substr(i + j + 1, j);
                if(t1 != t2 || f[i + 5])
                {
                    //可以作为后缀
                    ans.insert(t1);
                    f[i] = true;
                }
            }
        }
    
    cout << ans.size() << endl;
    //set自动按照字典序排序
    for(auto& t : ans)
        cout << t << endl;
    
    return 0;
    
}
```

分析问题多动笔，分类讨论化繁为简
如果分析到递推关系，就分类找递推式，考虑边界情况后按递推式实现代码