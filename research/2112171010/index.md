---
layout: post
type: research
date: 2021-12-17 10:10
category: MultiThread
title: Thread management in Python
subtitle: Shared memory, Server Process, Queue, Pipe
writer: 100
post-header: true
header-img: img/shared_memory.png
hash-tag: [python, multi_thread]
use_math: true
---

# Thread management in Python    
> Author: [SungwookLE](joker1251@naver.com)  
> Date  : '21.12/17  
>> 1. Shared Memory
>> 2. Server Process
>> 3. Queue
>> 4. Pipe


## 1. Shared Memory
### 1-1. 기본 사용법
- Shared memory란 두 프로세스 간에 특정 메모리를 공유하는 방식입니다. 두 프로세스는 해당 메모리에 접근하여 read/write할 수 있습니다.

```python
import multiprocessing
import time
  
def count_Time(int_Time_counter):
    while(True):
        int_Time_counter.value = int_Time_counter.value + 1
        print("Clock activated ...")
        time.sleep(1)

def print_Time(int_Time_counter):
    while(int_Time_counter.value < 20):
        print("Time: " + str(int_Time_counter.value))
        time.sleep(1)
  
if __name__ == "__main__":
    int_Time_counter = multiprocessing.Value('i',0)

    process_Count_time = multiprocessing.Process(target=count_Time, args=(int_Time_counter,))
    process_Print_time = multiprocessing.Process(target=print_Time, args=(int_Time_counter,))

    process_Count_time.start()
    process_Print_time.start()
```
- 위 코드에는 시간을 세는 process_Count_time과 시간을 출력하는 process_Print_time이라는 두 가지 종류의 프로세스가 존재합니다.
- 두 프로세스는 int_Time_counter라는 변수를 서로 공유합니다.
- 해당 함수 실행 결과는 아래와 같습니다.

```python
Clock activated ...
Time: 1
Clock activated ...
Time: 2
Clock activated ...
Time: 3
Clock activated ...
Time: 4
Clock activated ...
Time: 5
Clock activated ...
Time: 6
Clock activated ...
Time: 7
Clock activated ...
Time: 8
Clock activated ...
Time: 9
Clock activated ...
Time: 10
Clock activated ...
Time: 11
Clock activated ...
Time: 12
Clock activated ...
Time: 13
Clock activated ...
Time: 14
Clock activated ...
Time: 15
Clock activated ...
Time: 16
Clock activated ...
Time: 17
Clock activated ...
Time: 18
Clock activated ...
Time: 19
Clock activated ...
Clock activated ...
```
### 1-2. 한계
- 하지만 shared memory는 동시에 데이터 접근이 가능하며 이러한 특성은 다양한 문제를 발생시킬 수 있습니다. 

```python
import multiprocessing
from multiprocessing import process
import time
  
def count_Time(int_Time_counter):
    while(True):
        int_Time_counter.value = int_Time_counter.value + 1
        print("Clock activated ...")
        time.sleep(1)

def print_Time(int_Time_counter):
    while(True):
        print("Time: " + str(int_Time_counter.value))
        time.sleep(1)
  
if __name__ == "__main__":
    int_Time_counter = multiprocessing.Value('i',0)

    list_Process = []

    for i in range(10):
        process_Count_time = multiprocessing.Process(target=count_Time, args=(int_Time_counter,))
        list_Process.append(process_Count_time)
        
    process_Print_time = multiprocessing.Process(target=print_Time, args=(int_Time_counter,))

    for i in range(10):
        list_Process[i].start()
    process_Print_time.start()
```

- 위 코드는 열개의 process_Count_time과 한 개의 process_Print_time으로 구성됩니다.
- 만약 shared memory가 동시에 한 프로세스에서만 접근이 가능하다면 매 1초마다 print되는 int_Time_counter는 10씩 늘어나야합니다.
- 실제 실행결과는 아래와 같습니다.

```python
Clock activated ...
Clock activated ...
Clock activated ...
Clock activated ...
Clock activated ...
Clock activated ...
Clock activated ...
Clock activated ...
Clock activated ...
Clock activated ...
Time: 10
Clock activated ...
Clock activated ...
Clock activated ...
Clock activated ...
Clock activated ...
Clock activated ...
Clock activated ...
Clock activated ...
Clock activated ...
Clock activated ...
Time: 18
Clock activated ...
Clock activated ...
Clock activated ...
Clock activated ...
Clock activated ...
Clock activated ...
Clock activated ...
Clock activated ...
Clock activated ...
Clock activated ...
Time: 25
Clock activated ...
Clock activated ...
Clock activated ...
Clock activated ...
Clock activated ...
Clock activated ...
Clock activated ...
Clock activated ...
Clock activated ...
Clock activated ...
Time: 32
```

- process_Count_time은 실제로 1초에 10번 수행되기는 합니다. 하지만  1초마다 증가하는 숫자는 10은 아니며, 이는 동시에 shared memory에 접근할 수 있기 때문입니다.

### 1-3. 결론
- 결론적으로 share memory는 코딩하기 매우 쉽습니다. 해당 변수에 접근하여 read/write하는 것도 굉장히 직관적입니다.

- 하지만, 동시에 write하는 경우 문제가 발생할 수 있으니, 동시에 여러 프로세스에서 연산을 해야하는 변수에는 사용하지 않는 것이 좋겠습니다.


## 2. Server Process

### 2-1. 기본 사용법
- Server process란 여러 process를 관리하는 server가 여러 프로세스에서 사용되는 자원을 관리하는 형태를 말합니다.

```python
import multiprocessing
import time
  
def count_Time(int_Time_counter):
    while(True):
        int_Time_counter.value = int_Time_counter.value + 1
        print("Clock activated ...")
        time.sleep(1)

def print_Time(int_Time_counter):
    while(True):
        print("Time: " + str(int_Time_counter.value))
        time.sleep(1)
  
if __name__ == "__main__":
    with multiprocessing.Manager() as manager:
        int_Time_counter = manager.Value(typecode=int,value=0)
        print("Initial value: " + str(int_Time_counter.value))

        process_Count_time = multiprocessing.Process(target=count_Time, args = (int_Time_counter,))
        process_Print_time = multiprocessing.Process(target=print_Time, args = (int_Time_counter,))

        process_Count_time.start()
        process_Count_time.join()

        process_Print_time.start()
        process_Print_time.join()
```

- 위의 코드를 보면 with 문 안에서 process_Count_time, process_Print_time 두 프로세스를 manager가 관리합니다.
- int_Time_counter라는 manager관리 하의 공용으로 사용되는 object가 선언되고 이를 두 프로세스에서 사용합니다.
- 아래는 수행 결과입니다.

```python
Initial value: 0
Clock activated ...
Clock activated ...
Clock activated ...
Clock activated ...
Clock activated ...
Clock activated ...
Clock activated ...
Clock activated ...
```

- 근데 왜 process_Count_time만 동작할까요?
    - process.join()은 start()가 끝날 때까지 block 상태로 대기합니다.
    - 따라서, 아래 처럼 코드 변경해야합니다.

```python
import multiprocessing
import time
  
def count_Time(int_Time_counter):
    while(True):
        int_Time_counter.value = int_Time_counter.value + 1
        print("Clock activated ...")
        time.sleep(1)

def print_Time(int_Time_counter):
    while(True):
        print("Time: " + str(int_Time_counter.value))
        time.sleep(1)
  
if __name__ == "__main__":
    with multiprocessing.Manager() as manager:
        int_Time_counter = manager.Value(typecode=int,value=0)
        print("Initial value: " + str(int_Time_counter.value))

        process_Count_time = multiprocessing.Process(target=count_Time, args = (int_Time_counter,))
        process_Print_time = multiprocessing.Process(target=print_Time, args = (int_Time_counter,))

        process_Count_time.start()
        process_Print_time.start()

        process_Count_time.join()
        process_Print_time.join()
```

```python
Initial value: 0
Time: 0
Clock activated ...
Time: 1
Clock activated ...
Time: 2
Clock activated ...
Time: 3
Clock activated ...
Time: 4
Clock activated ...
```

## 3. Queue

### 3-1. 기본 사용법
- multiprocessing 시의 대표적인 communication 방식 중 하나 입니다.
- Queue에 메시지를 담은 후에 다른 프로세스에서 해당 메시지를 가져감으로써 두 프로세스 간에 메시지를 전달합니다.

```python
import multiprocessing
import time
  
def count_Time(int_Time_counter,queue_Time_counter):
    while (True):
        int_Time_counter = int_Time_counter + 1
        queue_Time_counter.put(int_Time_counter)
        print("Clock activated ...")
        time.sleep(1)

def print_Time(queue_Time_counter):
    while (True):
        if not queue_Time_counter.empty():
            print("Time: " + str(queue_Time_counter.get()))
  
if __name__ == "__main__":
    int_Time_counter = 0

    queue_Time_counter = multiprocessing.Queue()

    process_Count_time = multiprocessing.Process(target=count_Time, args = (int_Time_counter,queue_Time_counter))
    process_Print_time = multiprocessing.Process(target=print_Time, args = (queue_Time_counter,))

    process_Count_time.start()
    process_Print_time.start()
```

- 코드 수행결과는 아래와 같습니다.

```python
Clock activated ...
Time: 1
Clock activated ...
Time: 2
Clock activated ...
Time: 3
Clock activated ...
Time: 4
Clock activated ...
Time: 5
Clock activated ...
Time: 6
Clock activated ...
Time: 7
Clock activated ...
Time: 8
Clock activated ...
Time: 9
Clock activated ...
Time: 10
```

- 매초마다 clock이 동작하고 이를 출력하는 두 프로세스가 모두 잘 동작합니다.

### 3-1. 장점
- queue를 사용하는 방식은 두 프로세스간의 동기화를 보장합니다. 따라서, thread 및 process를 쓸 때 가장 안전합니다.

## 4. Pipe

### 4-1. 기본 사용법
- pipe는 두 노드 간의 단방향 통신입니다. sender와 receiver를 지정하여 한 방향으로의 변수를 전달합니다.

```python
import multiprocessing
import time
  
def count_Time(int_Time_counter,conn):
    while (True):
        int_Time_counter = int_Time_counter + 1
        conn.send(int_Time_counter)
        print("Clock activated ...")
        time.sleep(1)

def print_Time(conn):
    while (True):
        int_Time_counter = conn.recv()
        print("Time: " + str(int_Time_counter))
  
if __name__ == "__main__":
    int_Time_counter = 0

    conn_Sender, conn_Receiver = multiprocessing.Pipe()

    process_Count_time = multiprocessing.Process(target=count_Time, args = (int_Time_counter,conn_Sender))
    process_Print_time = multiprocessing.Process(target=print_Time, args = (conn_Receiver,))

    process_Count_time.start()
    process_Print_time.start()
```

- 위 코드에서 process_Count_time은 매초마다 int_Time_counter에 1을 더해서 pipe 통신으로 해당 변수를 송신합니다.

- process_Print_time은 pipe 라인을 통해 변수 값을 수신한 뒤에 프린트합니다.

- 결과는 아래와 같습니다.

```python
Clock activated ...
Time: 1
Clock activated ...
Time: 2
Clock activated ...
Time: 3
Clock activated ...
Time: 4
Clock activated ...
Time: 5
Clock activated ...
Time: 6
Clock activated ...
Time: 7
Clock activated ...
Time: 8
Clock activated ...
Time: 9
Clock activated ...
Time: 10
```
- 조금 희안한 것은 print_Time 함수에서 while() 조건을 따로 걸지 않고 true로 두어도 pipe 통신으로 변수가 송신되었을 때만 호출된다는 것
    - `recv()`
    연결의 반대편 끝에서 `send()`로 보낸 객체를 반환합니다. 뭔가 수신할 때 까지 블록합니다. 수신할 내용이 없고 반대편 끝이 닫혔으면 `EOFError`를 발생시킵니다.

### 4-1. 주의점
- pipe 통신 시 두 프로세스가 같은 pipe라인의 변수를 동시에 읽으려하거나 쓰려고 하는 경우 변수가 오염될 수 있습니다.