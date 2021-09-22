---
title:  "Autonomous Stack Integration with ROS"
excerpt: "ROS를 이용하여 Autonomous Stack을 Integration 해보자"

categories:
  - research
tags:
  - research

toc: true
toc_sticky: true
use_math: true
 
date: 2021-09-22
---

# Autonomous Integration Project: Step1
> AUTHOR: Sungwook LE  
> DATE: '21.9/22
> Lecture: [System Integration, Udacity](https://classroom.udacity.com/nanodegrees/nd013/parts/b9040951-b43f-4dd3-8b16-76e7b52f4d9d/modules/85ece059-1351-4599-bb2c-0095d6534c8c/lessons/01cf7801-7665-4dc5-a800-2a9cca06b38b/concepts/f48e03e9-3b2b-4395-9ead-595f4fbc7b79)    
> My Repo: [Here](https://github.com/SungwookLE/udacity_CarND_capstone)   

## 1. Introduction

- Autonomous **Full-Stack** 이란?
    - 자율주행 SW에서 `인지(Perception), 판단(Decision), 측위(Localization), 지도(Mapping), 계획(Planning), 제어(Control) 등` *Fully Autonomous Vehicle* 구현을 위해 필요한 기술(기능) 전체를 말하는 것
    ![fullstack1](/assets/full_stack1.png)  
    ![fullstack2](/assets/full_stack2.png)  
    - **ROS**와 같은 `Operating System` 위에서 `Framework`의 형태로 모듈/독립성을 가진 여러 기능SW의 `Stack`으로 자율주행이 구성된다는 의미(목표)에서 사용되는 단어

- `ROS`를 이용하여 구현해보는 것이 이번 포스팅의 목표이다.

## 2. ROS Memory
ROS를 익히며 배운 개념을 나만의 방식대로 정리해본다.

- `ROS` 구성 및 명령어
`rosnode, rostopic, rosmsg, roscore ...` 등을 하나씩 설명해보자.
- `rostopic list`: pub,sub 되고 있는 `topic`들의 list를 보여줌
- `rostopic info xxxx`: `xxxx`topic의 information을 알려줌. 예를 들면 `Type: styx_msgs/Lane`  
- `rosmsg info styx_msgs/Lane`: `styx_msgs/Lane`의 msg 구조를 알려줌

- `rospy.loginfo` 등 로그 메시지 [ROS.org](http://wiki.ros.org/rospy_tutorials/Tutorials/Logging)

1. `python`: `rospy`
    - `rospy.spin()` Vs. `rospy.rate(xxx), rate.sleep()`  
    `.spin()`은 어떤 메시지가 `Subs`들어오면 `callback`함수로 트리거 되서 작동되는 방식으로 해당 Node가 상호 기능 작동하게끔 하는 방식이고, `rospy.rate()와 rate.sleep()`을 쓰는 방식은 주기적으로 해당 Node를 작동하게 하는 방식이다. 두 개 방식을 하나의 Node에서 같이 써도 된다.  
    ```python
    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints:
                # Get closest waypoint
                closest_waypoint_idx = self.get_closest_waypoint_idx()
                self.publish_waypoints(closest_waypoint_idx)
            rate.sleep()
    ```




## 3. Project Brief
- 본 포스팅에선, ROS를 이용하여 자율주행 기능 `Stack`을 구현하고 Integration을 수행한다. 아래는 이번프로젝트에서의 `Node` 설명이다.
    - Including Stack: `traffic light detection`, `control`, and `waypoint following`
    - **Code Structure**
    ![structure](https://video.udacity-data.com/topher/2017/September/59b6d115_final-project-ros-graph-v2/final-project-ros-graph-v2.png)
    - `tl_detector`
    Traffic Light Detection Node를 포함하며, `/image_color`, `/current_pose`, `/base_waypoints` topics을 받아(subscribe) `/traffic_waypoint` topic을 publish한다.
    ![image](https://video.udacity-data.com/topher/2017/September/59b6d189_tl-detector-ros-graph/tl-detector-ros-graph.png)
    - `waypoint_updater`
    Waypoint Updater Node를 포함하며, obstacle 정보와 traffic light를 받아 target_velocity를 결정하는 역할을 한다. `/base_waypoint, /current_pose, /obstacle_waypoint, /traffic_waypoint` topic을 subscribe하고, list of waypoints와 target velocity를 `/final_waypoints`로 publish한다.  
    ![image](https://video.udacity-data.com/topher/2017/August/598d31bf_waypoint-updater-ros-graph/waypoint-updater-ros-graph.png)
    - `waypoint_loader`
    A package which loads the static waypoint data and publishes to /base_waypoints.
    - `waypoint_follower`
    A package containing code from Autoware which subscribes to `/final_waypoints` and publishes target vehicle linear and angular velocities in the form of twist commands to the `/twist_cmd topic`.
    - `twist_controller`
    DBW Node를 포함하며, PID controller로 `/current_velocity`, `/twist_cmd`, `vehicle/dbw_enabled` subscribe하여 `/vehicle/throttle_cmd, /vehicle/brake_cmd, /vehicle/steering_cmd` topics을 publish한다.
    ![image](https://video.udacity-data.com/topher/2017/August/598d32e7_dbw-node-ros-graph/dbw-node-ros-graph.png)
    - `styx, styx_msgs`
    provide a link between the simulator and ROS



## 여기까지 리뷰함 (9/22)
- [강의](https://classroom.udacity.com/nanodegrees/nd013/parts/b9040951-b43f-4dd3-8b16-76e7b52f4d9d/modules/85ece059-1351-4599-bb2c-0095d6534c8c/lessons/01cf7801-7665-4dc5-a800-2a9cca06b38b/concepts/fc053494-798d-4d29-b4c6-6d431d5c48db)
- Project: System Integration::7.DBW Walkthrough