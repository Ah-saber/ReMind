# DFS
# BFS

## 基本概念

- 宽度优先搜索（queue)
- 每次搜索，从分支开搜，一层接一层的进行搜索
- 要求所有边的权重相当

## 具体实现

利用队列实现

每次搜索时，队头出队，搜索队头的所有分支，满足条件就入队

多用于图的遍历，连通块的搜索

```C++


int dx[4] = {1, 0, -1, 0}, dy[4] = {0, -1, 0, 1};
bool d[N][N];
int a[x][y];

int q[N * N];


int bfs()
{
	int tt = 0, hh = 0;
	//队头初始化
	q[0] = a[x][y];

	while(hh <= tt)
	{
		//出队头
		auto t = q[hh ++];
		
		int ans = 0;
		for(int i = 0; i < 4; i ++)
		{
			int x = t.first + dx[i], y = t.second + dy[i];
			
			if(x >= 0 && x < n && y >= 0 && y < n && !d[x][y])
			{
				ans ++;
				q[++ tt] = {x, y};
				d[x][y] = true;
			}
		}
	}
	
	return ans;
}
```

## 最短路径

- 有时可以通过BFS来求最短路，要求边的权重是1，每次宽搜，第一次搜到就是最短路。

- DFS不可以，DFS第一次搜到时不一定是最短路