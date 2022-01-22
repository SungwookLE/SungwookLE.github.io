---
layout: post
type: algorithm
date: 2021-11-30 10:20
category: 코딩테스트 연습
title: 구명보트 Lv2
subtitle: 프로그래머스 그리디
writer: 100
hash-tag: [GREEDY, Programmers]
use_math: true
---


# 프로그래머스 > 그리디 > 구명보트
> AUTHOR: SungwookLE    
> DATE: '21.11/30  
>> PROBLEM: [문제링크](https://programmers.co.kr/learn/courses/30/lessons/42885)  
>> LEVEL: Lv2    

## 1. 시간 초과

- 아래처럼 풀었더니, 자꾸 시간초과가 나서 헤맸다..
- 직관적으로 조건에 만족하면 지우고, 만족하지 않으면 살려두고 하는 방식으로 O(N^2) 순회를 돌게 되니까.. 시간초과가 난 것이다.
- 혹시나, `erase` 같은 것 때문에 overhead time이 잡혀서 그런가 해서, for문을 써서 해봤지만 시간초과,,, 결국엔 알고리즘의 시간 초과를 해결해야만 했다.

```c++
int solution(vector<int> people, int limit) {
    int answer = 0;
    
    sort(people.begin(), people.end());
    
    while(people.size() > 1){
        
        bool flag = true;
        for(int i = people.size()-1; i > 0 ; --i){
            if (people[i] + people[0] <= limit){
                //cout << *people.begin() << " "<< people[i] << endl;
                people.erase(people.begin()+i);
                people.erase(people.begin());
                answer+=1;
                flag = false;
                break;
            }
        }
        if (flag){
            answer+=people.size();
            return answer;
        }
    }

    return answer+people.size();
}
```

## 2. 다른 사람의 풀이 참고
- 일종의, Binary Search 방식처럼 보이기도 하였다.
- `binary search` 방식은 아니고, 위의 시간초과난 알고리즘과 logic은 같은데, for문을 쓰지 않고, `sum<=limit`를 만족하지 않는 경우는 `right`를 왼쪽으로 당기고 `answer`를 하나 더해줌으로써 문제를 효율적으로 풀었다.

```c++
int solution(vector<int> people, int limit) {
    int answer = 0;
    
    sort(people.begin(), people.end());
    int left = 0;
    int right = people.size()-1;
    while ( left <= right ){
        int sum  =  people[left]+people[right];
        if(left != (right) && sum<=limit)
            left++;
        
        right -=1;
        answer+=1;
    }
   
    return answer;
}
```

- 이런 아이디어를 떠올리기란 참 쉽지 않은 일이다 ㅠㅠ..

## 끝