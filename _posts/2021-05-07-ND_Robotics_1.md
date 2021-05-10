---
title:  "Robotics: Chapter1-1"
excerpt: "Robotics Nanodegree: Introduction to Robotics"

categories:
  - research
tags:
  - research

toc: true
toc_sticky: true
 
date: 2021-05-07
---

### PROJECT1
> AUTHOR: SungwookLE  
> DATE: '21.5/7  

#### 프로젝트 치트시트라는데 참고  
 Download: [pdf](https://classroom.udacity.com/nanodegrees/nd209/parts/852e258d-b6c9-4823-b0af-0a7f77379583/modules/8a9ec5d0-dbd1-4f9b-80c5-c01a46aee151/lessons/962ef39a-4b29-4756-8dc1-05aca4075619/concepts/fdc36211-c805-4dde-b3b2-7b77624fe04d) 
 ![image](/assets/images/project1_description.png)  


#### 이번 프로젝트에서 시키는 건,  
 - Summary of Tasks  
  Let’s summarize what you should do in this project to create a simulation world for all your projects in this Robotics Software Engineer Nanodegree Program.  Build a single floor wall structure using the Building Editor tool in Gazebo. Apply at least one feature, one color, and optionally one texture to your   structure. Make sure there's enough space between the walls for a robot to navigate.  
  Model any object of your choice using the Model Editor tool in Gazebo. Your model links should be connected with joints.  
  Import your structure and two instances of your model inside an empty Gazebo World.  
  Import at least one model from the Gazebo online library and implement it in your existing Gazebo world.  
  Write a C++ World Plugin to interact with your world. Your code should display “Welcome to ’s World!” message as soon as you launch the Gazebo world file.  These tasks are just the basic requirements for you to pass the project! Feel free to have fun designing and importing multiple models.  
 - Rubrics(Requirements)
  HERE [LINK](https://review.udacity.com/#!/rubrics/2346/view) 

#### ASW VM 머신에서 아래와 같이 진행하였음 (5/10) 
* GAZEBO를 로컬에 설치해서 하려다가, 배보다 배꼽이 더 큰 기분이 들어,, 
  1. model 만들기   
   - gazebo 실행 -> model editor -> 내가 원하는 커스터마이즈드 로봇을 그려서 넣거나, 로봇을 넣고 저장  
   - gazebo 실행 -> building editor -> 내가 원하는 공간을 그려서 넣고 저장하면 됨  
   - 로봇과 공간을 폴더 트리로 저장해서 두고  
  2. world 만들기  
   - world 는 로봇과 공간을 저장하면 world 파일로 저장됨, 그니까 위에서 만든 로봇과 공간은 객체들이어서 어디던지 재조합하거나 끌어올수 잇음  
   - world는 픽스되는 스페셜한 한 공간임  
  3. script 폴더 만들고 cpp 파일 작성해서 넣기  
   - 사실 프로그래밍을 해야하는 나에게 이쪽 폴더가 제일 중요한건데 소스코드를 넣고 플로그인에 따라 인터렉티브하여 작동하면 됨

  Simple steps to interact with a World in Gazebo through a World plugin
  1- Create a directory for scripts inside “myrobot” to store a hello.cpp file
  ```
  $ cd /home/workspace/myrobot
  $ mkdir script
  $ cd script
  $ gedit hello.cpp
  ```
  Inside hello.cpp, include this code:

```c++
#include <gazebo/gazebo.hh>

namespace gazebo
{
  class WorldPluginMyRobot : public WorldPlugin
  {
    public: WorldPluginMyRobot() : WorldPlugin()
            {
              printf("Hello World!\n");
            }

    public: void Load(physics::WorldPtr _world, sdf::ElementPtr _sdf)
            {
            }
  };
  GZ_REGISTER_WORLD_PLUGIN(WorldPluginMyRobot)
}
```

  2- Create a CMakeLists.txt file
```
$ cd /home/workspace/myrobot
$ gedit CMakeLists.txt
```
Inside, CMakeLists.txt, include the following:
```
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

find_package(gazebo REQUIRED)
include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
list(APPEND CMAKE_CXX_FLAGS "${GAZEBO_CXX_FLAGS}")

add_library(hello SHARED script/hello.cpp)
target_link_libraries(hello ${GAZEBO_LIBRARIES})

```
  3- Create a build directory and compile the code
```
$ cd /home/workspace/myrobot
$ mkdir build
$ cd build/
$ cmake ../
$ make # You might get errors if your system is not up to date!
$ export GAZEBO_PLUGIN_PATH=${GAZEBO_PLUGIN_PATH}:/home/workspace/myrobot/build
```

4- Open your world file and attach the plugin to it
```
$ cd /home/workspace/myrobot/world/
$ gedit myworld
```

Copy this code
```
<plugin name="hello" filename="libhello.so"/>
```
and paste it under
```
<world name="default">
```
  5- Launch the world file in Gazebo to load both the world and the plugin
  ```
$ cd /home/workspace/myrobot/world/
$ gazebo myworld
```
6- Visualize the output
A Hello World! message is printed in the terminal. This message interacts with the Gazebo World that includes your two-wheeled robot.
Troubleshooting
In case your plugins failed to load, you'll have to check and troubleshoot your error. The best way to troubleshoot errors with Gazebo is to launch it with the verbose as such:
```
$ gazebo myworld --verbose
```


   4. 최종적으로는 아래와 같은 폴더트리로 gazebo 환경을 만들면 되고, 그렇게 제출하엿음

Directory Structure
The sample simulation world folder has the following directory structure:

    .Project1                          # Build My World Project 
    ├── model                          # Model files 
    │   ├── Building
    │   │   ├── model.config
    │   │   ├── model.sdf
    │   ├── HumanoidRobot
    │   │   ├── model.config
    │   │   ├── model.sdf
    ├── script                         # Gazebo World plugin C++ script      
    │   ├── welcome_message.cpp
    ├── world                          # Gazebo main World containing models 
    │   ├── UdacityOffice.world
    ├── CMakeLists.txt                 # Link libraries 
    └──                              

