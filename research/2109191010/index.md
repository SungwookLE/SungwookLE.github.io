---
layout: post
type: research
date: 2021-09-19 10:10
category: Path
title: Find Route- A Star Essential
subtitle: 주차 시스템에서 많이 쓰이는 경로계획법 A-star를 리뷰하자
writer: 100
post-header: true
header-img: 
hash-tag: [PATH, AStar]
use_math: true
---


# A star algorithm Review
> AUTHOR: Sungwook LE  
> DATE: '21.9/19  
> Reference: [My Implementation](https://github.com/SungwookLE/ReND_Cpp_Astar/blob/master/readme.md)  
> Code: [My Code](https://github.com/SungwookLE/ReND_Cpp_Astar/tree/Review)

## 1. Introduction
- A star algorithm은 효율적인 최단거리 길찾기 알고리즘으로, autonomous parking system에서 쓰인다.
![image](https://video.udacity-data.com/topher/2019/August/5d4b1057_addtoopen/addtoopen.png)

- 위의 구조에서 `CellSort(), ExpandNeighbors(), CheckValidCell(), Heuristic(), AddToOpen()` method가 **A Star** 알고리즘의 전부이다.

- `CellSort()` 이름을 통해 전체 알고리즘의 대략적으로 설명해보자. `Cell`은 현재 지도의 좌표(2차원 격자)를 의미한다. `Cell`을 `Sort`한다는 의미는 `Cell`이 가지고 있는 어떤 **Cost Value**를 **정렬**한다는 의미이다. 

- `A*` 알고리즘의 **Cost Function**은  $f = g + h$ 이다. g는 출발지 부터 해당 cell까지의 경로 거리를 의미하고, h는 목적지로부터 해당 cell까지의 경로 거리를 의미한다. heuristic value인 `h`는 다양한 방식으로 변형이 가능하다.

**<center>$A^*$ 알고리즘은 갈 수 있는 경로의 Cell의 Cost Value를 계산하고 작은 Cell을 선택하면서 최단거리 경로를 탐색한다</center>**

- 앞에서 기술한 바와 같이, `A star`는 Cost Value와 Cost Value에 따른 Sort를 통한 선택이 가장 핵심이 되는 부분이다.
  - $Cost \space value, \space f = g + h$
  - 이번 구현에서는 heuristic value $h$를 `Manhatan distance`로 목적지로부터 타겟 `cell` 의 거리로 하였다.  
  ```c++
  int manhatan_dist(std::vector<int> target, std::vector<int> goal){
    int res;
    res = std::abs(goal[0]-target[0]) + std::abs(goal[1]-target[1]);
    return res;
  }
  ```
  - $g$는 출발점으로 부터 현재 cell까지의 발생한 이동 경로 거리를 의미한다.
  - 구해진 $cost \space value$를 기준으로 Cell을 Sorting하고 선택한다.


## 2. 구현(`C++`)

- 이번 구현에서는 `maze.txt`를 입력받아 지도로 사용하므로 전체 코드에는 `ReadBoardFile()`이 존재하고, 찾은 경로를 출력해주는 `PrintSolution()`가 있으나, `Search()`알고리즘에 포커싱하여 살펴보자
- 핵심이 되는 `Search` 알고리즘은 다음과 같다.

```c++
void A_star::initialize(){
        closed[start[0]][start[1]] = 1; // closed
        heauristic_calculate();
        
        x = start[0];
        y = start[1];

        g = 0;
        h = heuristic[x][y];
        f = g+h;
        opens.push_back({f,h,x,y});

        found = false;
        resign = false;
        count = 0;
}

std::vector<std::vector<std::string>> A_star::Search(){

    initialize();
    while ( found != true && resign != true){

        if (opens.size() == 0){
            resign =true;
            std::cout << "Fail to find the route!" << std::endl;
            break;
        }
        else{
            // CellSort() as Descending order
            std::sort(opens.begin(), opens.end(), [](std::vector<int> a , std::vector<int> b){
                if (a[0] > b[0])
                    return true;
                else
                    return false;
            });

            next = opens.back();
            opens.pop_back();

            x = next[2];
            y = next[3];
            f = next[0];
            h = next[1];

            expand[x][y] = count;
            count+=1;

            if(( x==goal[0]) && (y==goal[1])){
                found = true;
                // SAVE NAVIGATION
                navi[x][y] = "E";
                while (( x!=start[0]) || (y!=start[0])){
                    int x_ex = x - delta[info[x][y]][0];
                    int y_ex = y - delta[info[x][y]][1];

                    navi[x_ex][y_ex] = delta_name[info[x][y]];
                    x = x_ex;
                    y = y_ex;
                }
                navi[x][y] = "S";
            }
            else{
                // ExpandNegihbors()
                for (int i =0 ; i < delta.size() ; ++i){
                    int x2 = x + delta[i][0];
                    int y2 = y + delta[i][1];

                    // CheckValidCell()
                    if ((x2>=0) && (y2>=0) && (x2<grid.size()) && (y2<grid[0].size())){
                        if ((closed[x2][y2] == 0) && (grid[x2][y2] == 0)){
                            int g2 = g+ cost;
                            int h2 = heuristic[x2][y2];
                            int f2 = g2 + h2;
                            
                            // AddToOpen()
                            opens.push_back({f2,h2,x2,y2});
                            closed[x2][y2]=1;
                            info[x2][y2]=i;
                        }
                    }
                }
            }
        }
    }
    return navi;
}
```

### 2-1. Results

  ```
  MAZE is: 
    *    1    0    0    0    0    0    0
    0    1    0    1    1    1    1    0
    0    1    0    0    0    0    1    0
    0    1    1    0    1    0    1    0
    0    1    0    0    1    1    1    0
    0    0    0    1    1    0    0    *
  ** 1 is grid / 0 is aisle

  COST is: 
    S   -1   17   18   19   20   21   22
    1   -1   16   -1   -1   -1   -1   23
    2   -1   15   11   12   13   -1   24
    3   -1   -1   10   -1   14   -1   25
    4   -1    8    9   -1   -1   -1   26
    5    6    7   -1   -1   -1   -1   E
  ** Number is cost value(how many step: 27)

  NAVIGATOR is: 
    *    #    >    >    >    >    >    v
    v    #    ^    #    #    #    #    v
    v    #    ^    <              #    v
    v    #    #    ^    #         #    v
    v    #    >    ^    #    #    #    v
    >    >    ^    #    #              *
  ```

## 3. Conclusion
- A star 알고리즘은 아래 Flow를 기억하자
  - 시작지점과 끝지점 그리고 MAP을 받아 초기화한다.
  - Cell 마다 Heuristic Value를 계산해 둔다.
  - 시작지점부터 주변의 Cell을 Neighbor(`상,하,좌,우`)로 추가하고 Neighbor의 Cost Value(`f=g+h`)를 계산한다.
  - Cost Value가 계산된 Cell은 `visited` 표시를 하여 중복 계산이 안되게끔 한다.
  - 제일 작은 Cost Value를 갖는 Cell을 다음 이동 지점으로 고르고 주변 Cell을 탐색한다.
  - 더 이상 새로운 cell을 추가할 수 없을 땐, 길을 못 찾는단 의미이고 목적지에 도달하면 경로를 출력한다.
- Heuristic Value를 잘 선택하면 탐색 횟수를 줄일 수도 있을 것이고, 최악의 경우엔 모든 cell을 탐색해서라도 경로를 찾아내는 **A star** algorithm에 대해 리뷰하였다.

## 끝