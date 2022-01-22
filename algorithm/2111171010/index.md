---
layout: post
type: algorithm
date: 2021-11-17 10:10
category: 코딩테스트 연습
title: 완주하지 못한 선수 Lv1
subtitle: 프로그래머스 HASH
writer: 100
hash-tag: [HASH, Programmers]
use_math: true
---


# 프로그래머스 > 해쉬 > 완주하지 못한 선수
> AUTHOR: SungwookLE    
> DATE: '21.11/17  
>> PROBLEM: [문제링크](https://programmers.co.kr/learn/courses/30/lessons/42576)
>> LEVEL: Lv1

## 1. 효율적이지 못한 접근 법
- 은근히, LEVEL1임에도 불구하고 효율적이게 코드 짜는 아이디어가 잘 떠오르지 않는다. 
- 문제를 이렇게 풀어도 정답은 나온다. 그러나 효율성에서 좋지 않다.
- 그도 그럴 것이, 2개의 벡터 컨테이너 전부를 순회해야하만 하니 O(N2)의 시간복잡도를 갖게 된다.

```c++
string solution(vector<string> participant, vector<string> completion) {
    string answer = "";
 
    vector<int> check(participant.size(),0);
    
    while(!completion.empty()){        
        string person1 = completion.back();
        completion.pop_back();
        
        for(int i =0 ; i < participant.size() ; ++i){
            string person2 = participant[i];
            if ( (person1 == person2) && (check[i] == 0) ){
                check[i] = 1;
                break;
            }
        }
    }
    
    for (int i =0 ; i < check.size() ; ++i){
        if (check[i] == 0)
            answer = participant[i] ;
    }
        
    return answer;
}
```

## 2. 해쉬 아이디어에서 가져온 효율적인 알고리즘

```c++
string solution(vector<string> participant, vector<string> completion) {
    string answer = "";
 
    // 정렬을 하면 알파벳 순으로 정렬이 되는데, 

    sort(participant.begin(), participant.end());
    sort(completion.begin(), completion.end());
    
    for(int i =0 ; i < completion.size() ; ++i){
        
        // 정렬 순서가 다르다는 것은, 달라지는 순서에 participant가 없다는 뜻이고,
        if (participant[i] != completion[i]){
            answer = participant[i];
            return answer;
        }
    }
    // for문 끝까지 찾아지지 않으면, participant의 끝에 선수가 완주자 목록에 없다는 의미다.
    answer = participant.back();
    return answer;
}
```


## 끝


