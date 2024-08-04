---
layout: post
type: algorithm
date: 2021-07-27 10:10
category: Study
title: CODING TEST 2D the minist distance value 2261
subtitle: Tree structure & Divide and Conquer to reduce O(logN)
writer: 100
hash-tag: [Divide_Conquer, baekjoon]
use_math: true
---


# DataStructure: 2D the minist distance value #2261
AUTHOR: SungwookLE   
DATE: '21.7/27  
PROBLEM: [백준#2261](https://www.acmicpc.net/problem/2261)  
REFERENCE: [REF](https://dhpark-blog.tistory.com/entry/BOJ-2261-%EA%B0%80%EC%9E%A5-%EA%B0%80%EA%B9%8C%EC%9A%B4-%EB%91%90-%EC%A0%90)&
[반례모음](https://bingorithm.tistory.com/8)  

## 1. 2D 좌표의 최소값 구하기
이 문제에서 많이 해맷던게,, 계속 시간초과가 나오고 그래서였다.  
아 런타임에러도 많았는데, 그건 sort 할때
```c++
std::sort(v.begin(), v.end(), [](auto a, auto b){
    if( a.first > b.first)
        return ture;
    else
        return false;
})
```
이렇게 선언해주어야지, 만약 `if(a.first >=b.first)`로 선언을 해버리면 무한 뻉뻉이가 돌면서 런타임에러가 나는 것이었다.  
이걸 제외하면 시간초과의 늪에 갇혀있었는데 그 이유는,  
  
처음에는 분할 방식으로 아래와 같이 접근했었다.

```c++
// mat은 std::vector<std::pair<int,int>> mat 로 선언되었다.
void solver(int start, int end){
            if (start == end) return;

            int temp = distance(mat[start], mat[end]); 
            ans=std::min(ans, temp);

            int mid = (start+end)/2;
            near_comp(start, end, mid);
            //std::cout << mid << std::endl;
            solver(start, mid);
            solver(mid+1, end);

            return;
        }
```
문제는 왼쪽과 오른쪽 집단이 서로 최소값 비교가 불가해서 답이 제대로 나오지 않는다는 것이었다.  
즉, 중간 mid 인덱스 부분의 군집데이터에 절대최소값이 존재하게 될 경우 해를 찾지 못하는 문제가 있다. (반례 참고)  
![image](https://casterian.net/wp-content/uploads/2018/04/%EA%B0%80%EC%9E%A5-%EA%B0%80%EA%B9%8C%EC%9A%B4-%EB%91%90-%EC%A0%90-3-252x300.png)

따라서, 아래 2번 코드 처럼 중간에 겹치는 영역을 스페셜하게 처리해주고, 데이터 개수가 3tick 이하일때는 brute-force 방식으로 뻉뺑이 돌려서 찾는 방법으로 하여,
전체 수행횟수를 줄여야지, 시간초과의 늪에서 벗어날 수 있다.  

## 2. 코드
항상 새로워 ㅋㅋㅋ,, 후후  
dynamic programming, 백트래킹(recursion), DFS(깊이탐색) 등은 쉽지가 않네  
```c++
class solver_2261{
    public:
        solver_2261(std::vector<std::pair<int,int>> _mat){
            mat = _mat;
            std::sort(mat.begin(), mat.end(), [](auto a, auto b){
                if (a.first < b.first)
                    return true;
                else
                    return false;
            });
            size = mat.size();
        }

        int solver(int start, int end){

            int count = end-start+1;

            if(count <=3){
                for(int i = start; i < end; ++i){
                    for(int j = i+1; j <=end ; ++j){
                        ans = std::min(ans, distance(mat[i], mat[j]));
                    }
                }
            }
            else{
                int mid =(start+end)/2;
                int left = solver(start, mid-1);
                int right = solver(mid+1, end);

                ans = std::min(left, right);

                std::vector<std::pair<int, int>> temp_mat;
                temp_mat.push_back(mat[mid]);

                for(int i = mid-1; i>=start; --i){
                    if (distance({mat[mid].first, 0}, {mat[i].first,0}) >= ans ) break;
                    temp_mat.push_back(mat[i]);
                }
                for(int i = mid+1; i <= end; ++i){
                    if (distance({mat[mid].first, 0}, {mat[i].first,0}) >= ans ) break;
                    temp_mat.push_back(mat[i]);
                }
                std::sort(temp_mat.begin(), temp_mat.end(), [](auto a, auto b){
                if (a.second < b.second)
                    return true;
                else
                    return false;
                });

                for(int i = 0 ; i < temp_mat.size()-1; ++i){
                    for(int j = i +1 ; j < temp_mat.size() ; ++j){
                        if (distance({temp_mat[i].second,0}, {temp_mat[j].second,0}) >= ans) break;
                        ans = std::min(ans, distance(temp_mat[i], temp_mat[j]));
                    }
                }
            }
            return ans;
        }
        int distance(std::pair<int, int> a, std::pair<int, int> b){
            return (a.first - b.first) * (a.first - b.first) + (a.second - b.second) * (a.second - b.second); 
        }
        void show_ans(){
            std::cout << ans << std::endl;
        }
        void show_mat(){
            std::cout << "MAT: \n";
            for(auto a : mat){
                std::cout << a.first << " " << a.second << "\n";
            }
        }

    private:
        std::vector<std::pair<int, int>> mat;
        int size = mat.size();
        int ans=1000000000;
};

int main(){
    int n;
    int x,y;

    std::cin >> n;
    std::vector<std::pair<int, int>> mat;

    for (int i=0 ; i < n ; ++i){
        std::cin >> x >> y;
        mat.push_back(std::make_pair(x,y));
    }

    solver_2261 solver(mat);
    solver.solver(0, n-1);
    solver.show_ans();

    return 0;
}
```

## 끝 
