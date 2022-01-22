---
layout: post
type: algorithm
date: 2021-11-22 10:10
category: 코딩테스트 연습
title: 가장 큰 수 Lv2
subtitle: 프로그래머스 정렬
writer: 100
hash-tag: [SORT, Programmers]
use_math: true
---



# 프로그래머스 > 정렬 > 가장 큰 수 
> AUTHOR: SungwookLE    
> DATE: '21.11/22  
>> PROBLEM: [문제링크](https://programmers.co.kr/learn/courses/30/lessons/42746)  
>> LEVEL: Lv2    

## 1. 잘못된 풀이1
- 아,, 이 문제에 거의 한시간 이상을 고민했는데 못 풀었다... 간단한 해결책이 있음을 알고 조금 허무했다.
- 시도해본것은 처음에는 backtracking으로 `full search` 를 해보았고 시간초과가 났다.

```c++
// 1. back_tracking으로 full search 해서 풀었는데, 시간초과 났음

void back_tracking(int iter, int n, vector<string> nums, vector<int> check, string ret, vector<string>& rets){
    
    if ( n == iter){
        rets.push_back(ret);
        return;
    }
    else{
        for(int i = 0; i < nums.size() ; ++i){
            if (check[i] == 0){
                check[i] = 1;
                ret+=nums[i];
                back_tracking(iter+1, n, nums, check, ret, rets);
                ret.erase(ret.length()-nums[i].length(), ret.length());
                check[i] = 0;
            }
        }
    }
}


string solution(vector<int> numbers) {
    string answer = "";
    
    int iter = 0, n = numbers.size();
    string ret;
    vector<string> rets;
    vector<string> nums;
    vector<int> check(n,0);
    
    for(int i = 0 ; i < numbers.size(); ++i)
        nums.push_back(to_string(numbers[i]));
    
    back_tracking(iter, n, nums, check, ret, rets);
    sort(rets.begin(), rets.end());
    answer = rets.back();
    
    return answer;
}
```


## 2. 잘못된 풀이2
- number를 string으로 바꾸어 저장한다음 `sort`를 이용해서 문제를 풀려고 했다.
```c++
sort(nums.begin(), nums.end(), [](string a, string b){
    for(int i =0 ; i < min(a.length(), b.length()); ++i){
        if ( a[i] > b[i])
            return true;
        else
            return false;
    }
    //...
});
```
- 이런 식으로 풀려고 했다. 정밀채점에 부분 테스트케이스가 틀려, 부분점수가 나왔고 이것 때문에 길게 고민했으나 답을 못 찾앗다 ㅠㅠ

## 3. 명쾌한 해답 (다른 사람의 풀이 참고)
- `sort` 함수 안에 compare 람다 함수를 보아라... `if (a+b) > (b+a) return true;`
- 간단하면서 명확하다..... 그래서 정답이다.
- 아이디어도 중요하다..

```c++
string solution(vector<int> numbers) {
    string answer = "";
    
    vector<string> nums;
    for(auto num: numbers)
        nums.push_back(to_string(num));
    
    
    sort(nums.begin(), nums.end(), [](auto a, auto b){
       
        if (a+b > b+a)
            return true;
        else
            return false;
     
    });
    
    for(int i =0 ; i < nums.size() ; ++i)
       answer += nums[i];
    
    if (answer[0] == '0')
        answer = "0";
    
    return answer;
}
```

## 끝

