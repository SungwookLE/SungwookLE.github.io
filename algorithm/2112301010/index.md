---
layout: post
type: algorithm
date: 2021-12-30 10:10
category: 코딩테스트 연습
title: 다시 풀어본, 정수삼각형 Lv3
subtitle: 프로그래머스 동적계획법
writer: 100
hash-tag: [Dynamic_Programming, Programmers]
use_math: true
---


# 프로그래머스 > 동적계획법 > 다시 풀어본 > 정수삼각형  
> AUTHOR: SungwookLE    
> DATE: '21.12/30  
>> PROBLEM: [문제링크](https://programmers.co.kr/learn/courses/30/lessons/43105)  
>> LEVEL: Lv3    

## 1. 직접 풀기 실패 ㅠㅠ.
- 다시 풀어본 것이나 역시나, 동일한 접근법으로 문제를 풀려고 하였고 시간초과 에러가 나면서 틀렸따.
- 브루트포스 방식으로 모든 경우의 수를 다 체크해서 더해서 최대 합인 경로를 찾아내는 방식으로 코드를 작성하였는데, 다른 코드를 참고해보니, 영리한(?) 방식을 적용해서 풀어야 시간초과가 나지 않는다.
- 오랜만에, c++를 쓰는 것이라 클래스를 활용해서 깔끔하게 풀려고 했는데,
시간초과를 해결하려고 이것저것 미봉책을 가져다 쓰다 보니, 코드가 깔끔하지도 않고 시간초과는 발생하였다.

```c++
class solver{
  public:
    solver(vector<vector<int>> triangle_): triangle(triangle_) {height = triangle.size(); }
    
    void dp(int iter, vector<int> path, int last_col, int sum, int& answer){
        if (iter == height){
            path_all.push_back(path);
            if (sum > answer)
                answer = sum;
        }
        else{
            
            for(int col =0 ; col < triangle[iter].size() ; ++col){
                if ( col >=last_col && col <= last_col+1){
                    path.push_back(triangle[iter][col]);
                    sum += triangle[iter][col];
                    dp(iter+1, path, col, sum, answer);
                    sum -= triangle[iter][col];
                    path.pop_back();
                }
            }
        }
    }
    
    void debug(){
        for(auto p : path_all){
            for(auto a : p)
                cout << a << " " ;
            cout << endl ;
        }
    }
  
  private:
    vector<vector<int>> triangle;
    vector<vector<int>> path_all;
    int height;
    
};

int solution(vector<vector<int>> triangle) {
    int answer = 0;
    
    solver solve(triangle);
    int iter =0;
    int last_col=0;
    int sum =0;  
    vector<int> path;
    
    solve.dp(iter, path, last_col, sum, answer);
    //solve.debug();
    
    return answer;
}
```

## 2. 다른 사람의 코드 참고
- `bottom-up` 방식으로 접근하여 모든 경로를 체크하지 않아도,
최대 합이 나오는 경로를 효율적으로 추려서 선택할 수 있다.
- `for문` 안을 보면 삼각형 밑에 `row`중 큰 녀석을 위에 `row`에 더해줌으로써 최대합이 나오는 경로를 효율적으로 추려서 찾아낼 수 있다.
- 이렇게 풀면 당연히, 시간 초과가 나지 않는다..

```c++
int solution(vector<vector<int>> t) {
    int answer = 0;

    for (int i = t.size() - 1; i > 0 ; i--)
    {
        for (int j = 0; j < t[i].size() - 1; j++)
        {
            if (t[i][j] > t[i][j + 1])
            {
                t[i - 1][j] += t[i][j];
            }
            else
            {
                t[i - 1][j] += t[i][j + 1];
            }
        }
    }
    //bottom-up으로 가면 바로 풀리네 ㅋㅋ,,
    answer = t[0][0];

    return answer;
}
```

## 끝