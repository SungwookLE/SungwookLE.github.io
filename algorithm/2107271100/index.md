---
layout: post
type: algorithm
date: 2021-07-27 11:00
category: Study
title: CODING TEST  Priority Queue 11279
subtitle: Priority Queue ... 
writer: 100
hash-tag: [priority_queue, baekjoon]
use_math: true
---


# 우선순위 큐 (Priority Queue)
- Author: SungwookLE
- DATE: '21.8/10
- BAEKJOON: [#11279](https://www.acmicpc.net/problem/11279)
- REFERENCE: [REF](https://junstar92.tistory.com/63)

## 1. 우선순위 큐
- First In, First Out의 일반적인 자료 컨테이너에서, 사용자가 우선순위에 따라 입력받은 데이터를 Queue에 저장하고 push, pop, top 등의 멤버함수를 통해 데이터를 핸들링할 수 있는 컨테이너를 우선순위 큐라고 한다. 이진트리로 구성하는 것이 FM 방식(손으로 구현할 때 제일 빠른)이 된다.  
[Figure1. 이진트리 예시(큰 숫자가 우선순위가 높은 우선순위 큐)]  
![Figure1](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FmL721%2FbtqH3BDU9cP%2F3scCDM3pn76QKo92Q4Wrb1%2Fimg.png)

- 그림에서와 같이 새로운 데이터가 입력되었을 때, 예시의 자료구조는 숫자가 클수록 우선순위가 높으므로 `25`라는 숫자는 child->parent로 올라가게 된다.  
- 우선순위 큐를 꼭 이진 트리(Tree) 방식으로 구현해야하는 것은 아니나, 시간복잡도 측면에서 유리하기 때문에 STL은 트리 방식을 따른다. 이 외에도 LIST(어레이)를 이용하여 우선순위에 따라 데이터를 저장하고 출력할 수 있으나 시간 복잡도에서 불리하긴 하다.
- 직접 트리 방식으로 구현하여도 백준 제출결과 시간초과가 나왔는데, 이는 메모리를 다루는 과정에서 비효율이 있었다고 예상된다. 역시 STL,, ㅠㅠ OTL,,,  

## 1-1. STL <QUEUE> 를 이용한 구현
- 딱히, 코멘트가 필요없는 것이 제공하는 `std::priority_queue`를 사용하였다.  
- 이런, STL 포맷의 데이터 컨테이너는 장점이 보통 `push, top, pop, empty`등의 기본 멤버함수를 가지고 있기 때문에 사용상에 불편함이 없다.  
```c++
// 우선순위 큐
#include <queue>
class solver_11279{
    public: 
    void solver(){
        std::cin >> N;
        int x;
        for (int i =0 ; i < N ; ++i){
            std::cin >> x;
            if (x==0){
                if (arr.empty())
                    std::cout << "0\n"; 
                else{
                    std::cout << arr.top() << "\n";
                    arr.pop();
                }
            }
            else
                arr.push(x);
        }
    }


    private:
        int N;
        std::priority_queue<int> arr;
};
```

## 1-2. LIST를 이용한 구현
- 동적 배열 할당을 통해 배열에 데이터를 넣고, `std::sort`를 이용하여 데이터를 우선순위에 따라 정렬한 후, 출력하는 방식이다.
- 출력된 결과물만 놓고 본다면야 결과는 같겠지만, 조금의 비효율이 존재하는 것이 데이터 전체를 sort 해야하는 과정에서(반복적으로) 시간 복잡도가 증가한다. `std::sort`는 quick_sort 등의 빠른 알고리즘을 사용하겠지만 몇번의 계산 회수는 더 필요할 것으로 예상이 된다.
- 기본적인 queue를 손으로 구현하는 방식에서 `top` 할 때, `sort`를 수행함으로써 우선순위가 제일 높은 것이 위에 오게끔 처리하였다.  

```c++
class prior_queue{
    public:
    prior_queue(){
        size = 0;
        queue_arr = new int[1]{0,};
    }

    prior_queue(int n): size(n){
        queue_arr = new int[size]{0,};
    }

    bool empty(){
        if (size ==0)
            return true;
        else
            return false;
    }
    void push(int x){
        size +=1;
        resize(queue_arr, size);
        queue_arr[size-1] = x;
    }

    void pop(){
        size -=1;
        if (size != 0)
            resize(queue_arr, size);
    }
    int top(){
        sort_queue();
        return queue_arr[size-1];
    }

    private:
    int* queue_arr;
    int size=0;
    void resize(int* &arr, int new_N){
        int *new_arr = new int[new_N];
        for(int i = 0 ; i < std::min(int(size), new_N) ; ++i)
            new_arr[i] = arr[i];
        delete [] arr;
        arr = new_arr;
    }
    void sort_queue(){
        std::sort(queue_arr, queue_arr+size, [](int a, int b){
            if ( a< b)
                return true;
            else
                return false;
        });
    }

};

class solver_11279{
    public: 
    solver_11279(){
        queue = new prior_queue;
    }
    void solver(){
        int N ;
        std::cin >> N;

        int x;
        for (int i =0 ; i < N ; ++i){
            std::cin >> x;
            if ( x == 0){
                if (queue->empty())
                    std::cout << "0\n";
                else{
                    std::cout << queue->top() <<"\n";
                    queue->pop();
                }
            }
            else
                queue->push(x);
        }
    }

    private:
        int N;
        prior_queue* queue;
        std::vector<int> ans;
};
```

## 1-3. TREE를 이용한 구현 (***중요)
- 이번 포스팅을 작성하는 이유이다.
- 트리에 잘 정리를 하면 뽑아 쓸때는 상당히 빠르다.
- 백엔드에서 데이터를 효율적으로 저장하고 관리하면, 프론트엔드에서는 데이터를 아주 빠르게 뽑아올 수 있다.
- 전체적인 시간복잡도도 LIST를 이용한 방식보다 우세하다.
- 레퍼런스로 [해당 블로그](https://junstar92.tistory.com/63)를 참고하였으니, 같이 보면 좋을 듯 하다.  
1) 데이터를 push 할 때,  
![fig2](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FmL721%2FbtqH3BDU9cP%2F3scCDM3pn76QKo92Q4Wrb1%2Fimg.png)  
25 라는 데이터가 입력되었고 child에서 parent로 더 큰수가 없을 때 까지 올라간다.

2) 데이터를 pop 할 때,  
해당 과정이 push보다는 조금더 복잡하다.  
![fig3](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbduNFL%2FbtqHR7dvKVK%2FCPjMJKrE3DKG66mxtdM6IK%2Fimg.png)  
78이 pop되고 나면 (b)와 같이 트리가 정리되어야 하는데, 이를 위해서 제일 `parent heapData[0]`에 제일 child `heapData[--n]`을 넣고 제 자리를 찾아갈 때 까지 반복문을 수행한다.

- 코드를 살펴보면 감을 잡을 수 있다. 익히는 것은 반복적으로 봐주어야할 듯 싶다.  
- 이진 트리에서 왼쪽부터 데이터를 채워나간다.  
![fig4](https://blog.kakaocdn.net/dn/cm41o7/btqyr6HY8bh/kFiWfuw1ShtTYiiEuGK1LK/img.png)
해당 그림은 tree 에 데이터를 원하는 형태로 만들어두고 query로 출력하는 방식의 예제에서 따온 그림이다.  

```c++ 
template<typename T>
class Heap{
public:
    Heap(int maxSize = 100): n(0), maxSize(maxSize), heapData(new T[0]) {}
    ~Heap(){ delete[] heapData; }

    void push(const T data){
        resize(heapData, n+1);
        heapData[n] = data;
        int parent = (n-1)/2;
        int child = n;

        while (parent >= 0 && heapData[parent] < heapData[child]){

            T tmp = heapData[parent];
            heapData[parent] = heapData[child];
            heapData[child] = tmp;

            child = parent;
            parent = (child-1)/2;
        }
        ++n;
    }

    void pop(){
        if (!empty()){
            heapData[0] = heapData[--n];
            resize(heapData, n);

            int parent = 0;
            int child = parent*2 +1;
            bool placed = false;

            while (!placed && child < n){
                if (child < (n-1) && heapData[child] < heapData[child+1] )
                    child+=1;

                if (heapData[parent] >= heapData[child])
                    placed = true;
                else{
                    T tmp = heapData[parent];
                    heapData[parent] = heapData[child];
                    heapData[child] = tmp;
                }

                parent = child;
                child = parent*2 +1;
            }
        }
        else
            std::cout << "EMPTY!\n";

    }

    T top(){
        if (!empty())   
            return heapData[0];
        else
            return 0;
    }

    int size(){
        return n;
    }

    bool empty(){
        return (n==0);
    }

    void print(){
        std::cout << "[";
        for(int i =0 ; i < n ; ++i)
            std::cout << heapData[i] << " ";
        std::cout <<"]";
    }

private:
    int n;
    int maxSize;
    T* heapData;

    void resize(int* &arr, int new_N){
        int *new_arr = new int[new_N];
        for(int i = 0 ; i < std::min(n, new_N) ; ++i)
            new_arr[i] = arr[i];
        delete [] arr;
        arr = new_arr;
    }
};

template<typename T>
class solver_11279{
public:
    solver_11279(){
        heap = new Heap<T>;
    }
    ~solver_11279(){
        delete heap;
    }

    void solver(){
        std::cin >> N;
        int x;
        for(int i =0 ; i < N ; ++i){
            std::cin >> x;

            if (x==0){
                if (!heap->empty()){
                    std::cout<< heap->top() <<std::endl;
                    heap->pop();
                }
                else
                    std::cout << heap->top() <<std::endl;
            }
            else
                heap->push(x);
        }
    }

private:
    Heap<T>* heap;
    int N;
};
```
## 1-4. 마무리
- 트리 자료구조는 효율적이다!

## 끝
  
