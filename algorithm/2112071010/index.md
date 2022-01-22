---
layout: post
type: algorithm
date: 2021-12-07 10:10
category: 코딩테스트 연습
title: 정수 삼각형 Lv3
subtitle: 프로그래머스 동적계획법
writer: 100
hash-tag: [Dynamic_Programming, Programmers]
use_math: true
---



# 프로그래머스 > 동적계획법 > 정수 삼각형
> AUTHOR: SungwookLE    
> DATE: '21.12/07  
>> PROBLEM: [문제링크](https://programmers.co.kr/learn/courses/30/lessons/43105#)  
>> REFERENCE: [참고](https://velog.io/@skyepodium/%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%A8%B8%EC%8A%A4-%EC%A0%95%EC%88%98-%EC%82%BC%EA%B0%81%ED%98%95)  
>> LEVEL: Lv3    

## 1. 혼자서 푸는 것에 실패.. 근접 ㅠ
- `DP` 카테고리에 있으니까, 일단 `memoization`을 생각해서 무얼 담을까 고민하다가, 모든 중간 계산값을 다 더하고 마지막 `triangle`의 row에서 최대값을 판단하자고 마음을 먹었다.
- 근데, 이게 점화식을 만들어야하는데 규칙 찾기가 애매한 거다 ㅠㅠ...

```c++
/**
    * DP[0] = {7};                              = triangle[0][0];
    *
    * DP[1] = {10,15};                          = DP[0][0] + triangle[1][0] , DP[0][0] + triangle[1][1];
    *
    * DP[2] = {18, 11, 16, 15};                 = DP[1][0] + triangle[2][0] ,
    *						    DP[1][0] + triangle[2][1] , DP[1][1] + triangle[2][1] ,
    *						    DP[1][1] + triangle[2][2]
    *
    * DP[3] = {20, 25, 18, 23, 15, 20, 19, 19}; = DP[2][0] + trinagle[3][0] , 
    *						    DP[2][0] + triangle[3][1] , DP[2][1] + triangle[3][1] , DP[2][2] + triangle[3][1] ,
    *						    DP[2][1] + triangle[3][2] , DP[2][2] + triangle[3][2] , DP[2][3] + triangle[3][2] ,
    *						    DP[2][3] + triangle[3][3]
    *
    */
```

- `vector`컨테이너(변수명 DP)에 중간값을 어떻게 저장하고 꺼내 쓸까 생각하니까 3중 for문을 생각하게 되고 뭔가 꼬여가지고 시간 소모를 많이 했다..
- 다른 사람의 코드를 보니까, `그리디` 방식을 혼합해서 매우 간단하게 풀었더라.. 참고하여라..

## 2. 다른 사람 풀이 참고

- 그리디 방식과 DP방식의 짬뽕으로 풀면 매우 간단하다.
- triangle의 왼쪽과 오른쪽 중에 큰 쪽으로 선택해서 더해줘 나가면 되는 것이다. (`그리디`)
- 앞전에 계산된 `sum`값을 재활용하여 더해 나가면 되는 것이다. (`DP`)

```c++
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

using namespace std;

int solution(vector<vector<int>> triangle) {
    int answer = 0;
    
    //Memoization: DP문제는 큰문제를 작은 문제로 쪼개서 푸는 것으로,
    //작은 문제의 정답의 결과가 큰 문제의 중간답이 되어 최종 답을 얻어내는 과정으로 푼다.
    
    vector<vector<int>> DP(triangle.size(), vector<int> (501,0));
    DP[0] = triangle[0];
    
    if (triangle.size() < 2){
        answer = triangle[0][0];
    }
    else{
        for(int i = 1 ; i < triangle.size() ; ++i){
            for(int j = 0 ; j <=i; ++j){
                // 1) 삼각형 제일 왼쪽 끝인 경우
                if (j==0){
                    DP[i][j] = DP[i-1][j] + triangle[i][j];
                }            
                // 2) 삼각형 제일 오른쪽 끝인 경우
                else if (i==j){
                    DP[i][j] = DP[i-1][j-1] + triangle[i][j];
                }
                // 3) 삼각형 왼쪽, 오른쪽 끝인 아닌 내부인 경우
                else{
                    DP[i][j] = max(DP[i-1][j-1], DP[i-1][j]) + triangle[i][j];
                }
                // 최대값 갱신
                answer = max(answer, DP[i][j]);
            }
        }
    }
    return answer;
}
```

## 끝