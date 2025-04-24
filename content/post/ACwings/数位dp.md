例题：[338. 计数问题 - AcWing题库](https://www.acwing.com/problem/content/340/)

给定a，b两个数，求a和b之间的数字0~9各自出现的次数（每一位）
多组测试，每组范围1e8

直接找肯定超时

## 状态表示

这道题要转换一下思维
要求的是 a ， b 之间，各个数字出现的次数，并不好直接表示（因为一旦起始与结束的数字可变，就会有多种情况需要考虑），那就去掉起点，直接从0~n的计算可以用dp的思想表示出来
可以让集合为  **从 0 到 a(b) 的数中各位上各个数字出现的次数** 

属性是 数量

## 状态计算

表示之后，可以利用 到 b 出现的数量 - 到 a 出现的数量 得到 最终结果

分类计算：
- 例如计算 1 出现在第4位上的次数
1. 如果  ，那么不管第4位上是不是1，后面的位置取任意值都不会大于终点  可以取
2. 如果 中间数 前三位 等于 终点数 前三位，
	- 终点第4位小于1，那么这种情况下中间数第4位为1的情况是0
	- 终点第4位大于1，那么中间数就有第4位为1的情况存在，其之后的位置数字任取都不会大于 终点数
	- 终点第4为为1， 那么第4位以后的数就不可以大于终点的第4位以后的数
		![Pasted image 20240312223218|400](https://raw.githubusercontent.com/Ah-saber/MyPic/main/Pasted%20image%2020240312223218.png)

## 实现

实现其实不简单，要考虑到前缀零的特殊情况

同时，逆序操作会更加方便

```C++
#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;


//计算a在区间的数值
int get_num(vector<int> a, int l, int r)
{
    int res = 0;
    for(int i = l; i >= r; i --)
        res = res * 10 + a[i];
    
    return res;
}

int pow10(int n)
{
    int ans = 1;
    while(n --)
        ans *= 10;
        
    return ans;
}

int count(int a, int x)
{
    vector<int> s;
    //逆序a
    while(a)
    {
        s.push_back(a % 10);
        a /= 10;
    }
    int n = s.size();
    
    int ans = 0;
    //如果x为0，则要考虑前缀0，加上!x 来特判x=0的起始位置
    for(int i = n - 1 - !x; i >= 0; i --)
    {
        //如果i前面没有数，就没有所谓前面的数小于终点数了
        if(i < n - 1)
        {
            ans += get_num(s, n - 1, i + 1) * pow10(i);  // 第i位之后任取
            //如果x=0，不能有前缀0，最小必须是1，所以少一位
            if(!x)  ans -= pow10(i);
        }
        //如果当前位与所求字数x相同，则后面不能超过原来的数，但是从0开始，所以在原来的数基础上加一
        if(s[i] == x) ans += get_num(s, i - 1, 0) + 1;
        else if(s[i] > x) ans += pow10(i); //小于后面的任取（i位后面）0-99=100，所以传i
    }
    
    return ans;
}

int main()
{
    int a, b;
    
    while(cin >> a >> b, a || b)
    {
        if(a > b) swap(a, b);
        
        
        int ans;
        for(int i = 0; i < 10; i ++)
        {
            ans = count(b, i) - count(a - 1, i);
            cout << ans << ' ';
        }
        
        cout << endl;
    }
    
    return 0;
}
```
