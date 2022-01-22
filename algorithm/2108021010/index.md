---
layout: post
type: algorithm
date: 2021-08-02 10:10
category: Study
title: CODING TEST Binary Search Problem
subtitle: Binary Search for finding the biggest answer
writer: 100
hash-tag: [binary_search, baekjoon]
use_math: true
---


# DataStructure: Binary Search
AUTHOR: SungwookLE   
DATE: '21.8/2  
PROBLEM: [백준#1920](https://www.acmicpc.net/problem/1920), [백준#10816](https://www.acmicpc.net/problem/10816), [백준#1645](https://www.acmicpc.net/problem/1645), [백준#2805](https://www.acmicpc.net/problem/2805), [백준#2110](https://www.acmicpc.net/problem/2110)   

## 1. Binary Search 
- 계속 탐색 대상의 데이터를 반으로 줄여나가므로 이분탐색의 시간 복잡도는 O(logN)가 된다.  
- 주어진 배열에서 특정값이 존재하는지 찾는 방법으로, 먼저 배열을 오름차순(내림차순도 무방)으로 정렬하고, 반을 쪼개어 왼쪽/오른쪽 비교해나가며 루프를 돌면서 원하는 요소가 배열 내에 있는지 찾는 방식이다.   

## 2. 가장 기본적인 바이너리 서치
- 주의할 점은 while 루프가 제대로 종료되게 하기 위해서, start = mid+1 , end = mid -1 을 해주는 부분인데, +1 / -1을 해줌으로써 , 해가없는 경우에도 `start > end`  지점을 만들어 루프를 종료시킨다.  
```c++
void solver(){
        int target = array[i];
        int start =0;
        int end = array.size()-1;
        int mid;

        while(start <= end ){
            mid = (start+end)/2;

            if (array[mid] < target)
                start = mid+1;
            else if (array[mid] > target)
                end = mid-1;
            else
                break;
        }
        std::cout << "TARGET in " << mid <<" idx." << std::endl;         
    }
}  
```

## 3. 여러 가능한 해 중, 가장 큰 값을 찾기
- 백준에서는 이분탐색법으로 풀 수 있는 여러 문제를 소개하고 있고 그 중 하나가, 여러 가능한 해 중, 조건에 맞는 해를 고르는 문제이다.
- [백준#2805](https://www.acmicpc.net/problem/2805) 문제를 살펴보자, 문제는 적어도 M미터의 나무를 집에 가져가기 위해서 절단기에 설정할 수 있는 높이의 최댓값을 구하는 문제이다.  
그니까, 절단기 설정 높이는 여러개가 가능한데 그 중 가장 큰 값을 구하는 문제이다. 이러한 문제도 이분탐색법으로 풀 수 있다.    

```c++
void solver(){

    long long cut;
    long long remain = 0;

    long long start =0;
    long long end = trees.back();

    while (start <= end){

        cut = (start+end)/2;
        remain=0;

        for(int i = 0 ; i < trees.size() ; ++i){
            if (trees[i] > cut)
                remain = remain+ (trees[i] - cut);
        }    

        if (remain < M)
            end = cut-1;
        else if (remain >= M)
            start = cut+1;
    }
    std::cout << end << std::endl;
}
```
- 먼저, `start`와 `end`가 #2에서는 idx를 가르켰다면 여기에서는 값을 가리키고 있다.
- 둘째로, if문을 보면 따로 `break`문은 없고 `while`문이 `start`와 `end`가 엇갈릴 때 까지, 즉 해를 만족하는 경우에서의 범위(`start~end`)를 구한다.   
- 이 때, remain 이라는 특정 조건을 만족하는 range 중 가장 큰 값 `end`를 출력함으로써 여러 가능한 해 중 가장 큰 값을 뽑아내고 있다
```c++
if (remain < M)
    end = cut-1;
else if (remain >= M)
    start = cut+1;
```
- 이 부분에서 `else if`부분에서 `>=`가 있기 때문에 최대값이 뽑히는 것이고
- `if (remain <= M)` 로 한다면 remain과 M 이 값을때 점점더 작은 값으로 이동시키니까 최소값이 나옴
- [백준#2110](https://www.acmicpc.net/problem/2110)문제를 한번 더 풀어보는 것을 추천한다.  

  
여러번, 반복해서 보면서 패턴을 익히면 될 것 같음, 어려운 문제는 아니다.  

## 끝  



