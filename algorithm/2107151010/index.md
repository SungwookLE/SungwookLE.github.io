---
layout: post
type: algorithm
date: 2021-07-15 10:10
category: Study
title: CODING TEST Matrix Power Calculation 10830
subtitle: Divide and Conquer to reduce O(n)
writer: 100
hash-tag: [Divide_Conquer, baekjoon]
use_math: true
---


# DataStructure: Matrix Power Calculation #10830
AUTHOR: SungwookLE  
DATE: '21.7/15  
PROBLEM: [백준#10830](https://www.acmicpc.net/problem/10830)  
REFERENCE: https://ssungkang.tistory.com/entry/C-BAEKJOON-10830-%ED%96%89%EB%A0%AC-%EC%A0%9C%EA%B3%B1  

## 1. 분할정복
- 아래 코드 `matrixPow`를 읽어보자

## 2. CODE
```c++
std::vector<std::vector<long long int>> matrixMul(std::vector<std::vector<long long int>> A, std::vector<std::vector<long long int>> B){
        int n =A.size();
        std::vector<std::vector<long long int>> C(n, std::vector<long long int>(n));
            for(int row =0 ; row < n ; ++row){
                for(int col =0 ; col <n ; ++col){
                    for(int prod =0 ; prod < n ; ++prod)
                        C[row][col] += A[row][prod]*B[prod][col];

                    C[row][col] %= 1000;
                }
            }
        return C;
}

// 아래 매트릭스를 보면 재귀호출을 해서, 지수보다 더 적게 연산을 수행하게 됨 => 효율적

std::vector<std::vector<long long int>> matrixPow(std::vector<std::vector<long long int>> A, int pow){
        if (pow == 0 ) return ones;
        else if (pow == 1) return A;
        else if ( pow % 2 == 0){
            std::vector<std::vector<long long int>> temp = matrixPow(A, pow/2);
            return matrixMul(temp, temp);
        }
        else{
            std::vector<std::vector<long long int>> temp = matrixPow(A, pow-1);
            return matrixMul(temp, A);
        }
}
```

## 끝