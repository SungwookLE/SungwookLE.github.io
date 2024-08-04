---
layout: post
type: algorithm
date: 2021-07-25 10:10
category: Study
title: CODING TEST Segment Tree 6549
subtitle: Tree structure & Divide and Conquer to reduce O(logN)
writer: 100
hash-tag: [Divide_Conquer, baekjoon]
use_math: true
---


# DataStructure: Segment Tree Get highest histogram size #6549  
AUTHOR: SungwookLE  
DATE: '21.7/25  
PROBLEM: [백준#10830](https://www.acmicpc.net/problem/6549)  
REFERENCE: https://cocoon1787.tistory.com/314

## 1. 세그먼트 트리
- 먼저, 구간합을 구하는 일반적인 트리 코드를 살펴보자

## 2. 구간합 구하는 트리
- REF: https://blog.naver.com/ndb796/221282210534 구간합트리  

```c++

class segment_tree{
    public:
        // 트리 클래스
        segment_tree(std::vector<int> _arr){
            given_arr = _arr;
            tree.resize(4*given_arr.size());
        }
        int init(int start, int end, int node);
        int sum(int start, int end, int node, int left, int right);
        void show_tree(){
            std::cout << "TREE: \n";
            for(int i = 1 ; i < tree.size(); i=i*2+1){
                for(int j =(i-1)/2+1; j <= i ; ++j)
                    std::cout << tree[j] << " ";
                std::cout << std::endl;
            }
        }
        void show_given_arr(){
            std::cout<<"Given arr: \n";
            for(auto a : given_arr)
                std::cout << a<< " ";
            std::cout << std::endl;
        }
        void update(int start, int end, int node, int index, int dif);
    private:
    std::vector<int> tree ={-1};
    std::vector<int> given_arr;
};

int main(){
    int N;
    std::cin>> N;
    std::vector<int> given(N);
    for(int i =0 ; i < N ; ++i)
        std::cin >> given[i];

    segment_tree st(given);

    st.init(0, given.size()-1, 1);
    st.show_given_arr();
    st.show_tree();

    std::cout << "\nSUM: idx[1~2] \n";
    std::cout << st.sum(0, given.size()-1, 1, 1, 2) << std::endl;

    std::cout << "UPDATE: idx[1] to 10 \n";
    st.update(0, given.size()-1, 1, 1, 10);
    st.show_tree();
    
    std::cout << "\nSUM: idx[1~2] \n";
    std::cout << st.sum(0, given.size()-1, 1, 1, 2) << std::endl;

    return 0;
}

// start: 시작 인덱스, end: 끝 인덱스
int segment_tree::init(int start, int end, int node){
    if (start == end){
        tree[node] = given_arr[start];
        return tree[node];
    }

    int mid = (start+end)/2;
    // 재귀적으로 두 부분을 나눈 뒤에 그 합을 자기 자신으로 합니다.
    tree[node] = init(start, mid, node*2) + init(mid+1, end, node*2+1);

    return tree[node];
}
// start: 시작 인덱스, end: 끝 인덱스
// left, right: 구간 합을 구하고자 하는 범위
int segment_tree::sum(int start, int end, int node, int left, int right){
    //범위 밖에 있는 경우
    if (left > end || right < start)
        return 0;
    // 범위 안에 있는 경우
    if (left <= start && end <=right)
        return tree[node];
    //그렇지 않다면 두 부분으로 나누어 합을 구하기
    int mid = (start+end) / 2;
    return sum(start, mid, node*2, left, right) + sum(mid+1, end, node*2+1, left, right);
}

// start: 시작 인덱스, end: 끝 인덱스
// index: 구간 합을 수정하고자 하는 노드
// dif: 수정할 값
void segment_tree::update(int start, int end, int node, int index, int dif){
    
    //범위 밖에 있는 경우
    if(index < start || index > end) return;
    //범위 안에 있으면 내려가며 다른 원소도 갱신
    tree[node] += dif;
    if (start ==end) return;
    int mid = (start+end) / 2;
    update(start, mid, node*2, index, dif);
    update(mid+1, end, node*2+1, index, dif);
}
```

## 3. #6549 문제 접근
세그먼트 트리를 이용한 문제, 어렵네 @SungwookLE   
- 내가 원하는 답을 가지고 있는 트리를 구하고  
- 트리에서 액션을 하는 쿼리 함수를 짠다음에  
- 솔버에서 쿼리를 호출해서 O(logN) 의 복잡도로 함수를 짜는 것  

|||||||||
|---|---|---|---|---|---|---|---|
|INDEX|0|1|2|3|4|5|6|
|arr[]|2|1|4|5|1|3|3|

![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbGXuiQ%2FbtqRqbCuWOQ%2FzDdpziCsKNjUeA5pdT2KI1%2Fimg.png)


## 4. 코드
```c++

class solver_6549{

    public:
    solver_6549(std::vector<long long> _arr){
        arr = _arr;
        tree.resize(arr.size()*4);
    }

    int init(int start, int end, int node){

        if (start == end){
            tree[node] = start;
            return tree[node];
        }

        int mid = (start+end)/ 2;
        int left_index = init(start, mid, node*2);
        int right_index = init(mid+1, end, node*2+1);

        if (arr[left_index] < arr[right_index])
            tree[node] = left_index;
        else
            tree[node] = right_index;

        return tree[node];
    }

    int query(int start, int end, int node, int left, int right){

        if ( end < left || start > right) return -1;
        if ( start >= left && end <= right) return tree[node];

        int mid = (start+end)/2;
        int left_index = query(start, mid, node*2, left, right);
        int right_index = query(mid+1, end, node*2+1, left, right);

        if(left_index == -1) return right_index;
        else if (right_index == -1) return left_index;
        else{
            if ( arr[left_index] < arr[right_index])
                return left_index;
            else
                return right_index;
        }
    }

    void solve(int left, int right){
        if (left > right) return;

        int index = query(0, arr.size()-1, 1, left, right);
        ans = std::max(ans, arr[index] * (right-left+1) );

        //분할정복
        solve(left, index-1);
        solve(index+1, right);
    }

    void show_arr(){
        std::cout << "ARRAY: \n";
        for(int i: arr )
            std::cout << i << " " ;
        std::cout<<std::endl;
    }

    void show_tree(){
        std::cout << "TREE: \n";
        for(int i = 1 ; i < tree.size(); i=i*2+1){
            for(int j =(i-1)/2+1; j <= i ; ++j)
                std::cout << tree[j] << " ";
            std::cout << std::endl;
        }
    }

    long long ans=0;

    private:
    std::vector<long long> arr;
    std::vector<int> tree;

};

int main()
{
    int n;
    while (true)
    {
        std::cin >> n;
        if (n == 0)
            break;

        std::vector<long long> arr(n);
        for (int i = 0; i < n; i++)
            std::cin >> arr[i];
        
        solver_6549 solver(arr);
        solver.init(0, arr.size()-1, 1);

        solver.solve(0, arr.size()-1);
        std::cout << solver.ans << std::endl;

    }
    return 0;
}
```

## 끝