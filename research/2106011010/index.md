---
layout: post
type: research
date: 2021-06-01 10:10
category: ROS
title: Robotics Go Chase It
subtitle: 흰 공을 따라다니는 모바일 로봇(카메라)
writer: 100
post-header: true
header-img: /assets/images/GoChaseIt1.gif
hash-tag: [ROS, robot]
use_math: true
---

## PROJECT: Go Chase It
> AUTHOR: SungwookLE  
> DATE: '21.6/1  

### 1. 흰 공을 따라서 움직이는 모바일 로봇  
- ROS 환경에서 서비스, pub/sub, 클라이언트 <-> 센서데이터 등 환경 구성
- Gazebo 연계하여 시뮬레이션

### 2. source code
- drive_bot.cpp, process_image.cpp

```c++
// : process_image.cpp
#include "ros/ros.h"
#include "ball_chaser/DriveToTarget.h"
#include <sensor_msgs/Image.h>
#include "geometry_msgs/Twist.h"
#include <vector>

// Define a global client that can request services
ros::ServiceClient client;

// This function calls the command_robot service to drive the robot in the specified direction
void drive_robot(float lin_x, float ang_z)
{
    // TODO: Request a service and pass the velocities to it to drive the robot
    //ROS_INFO("Drive! linear_x: %1.2f, angular_z: %1.2f", lin_x, ang_z);
    geometry_msgs::Twist motor_command;
    motor_command.linear.x = lin_x;
    motor_command.angular.z = ang_z;

    ball_chaser::DriveToTarget srv;

    srv.request.linear_x = motor_command.linear.x;
    srv.request.angular_z = motor_command.angular.z;

    if (!client.call(srv))
        ROS_ERROR("Failed to call service drive_robot");
    
}

// This callback function continuously executes and reads the image data
void process_image_callback(const sensor_msgs::Image img)
{   

    int white_pixel = 255;

    // TODO: Loop through each pixel in the image and check if there's a bright white one
    // Then, identify if this pixel falls in the left, mid, or right side of the image
    // Depending on the white ball position, call the drive_bot function and pass velocities to it
    // Request a stop when there's no white ball seen by the camera

    int point_pixel=0;

    int step_left_size = img.step/3;
    int step_right_size = img.step*2/3;
    
    bool find_white_flag = false;
    std::vector<int> image_pixel_row(img.step,0);
    std::vector<std::vector<int>> image_pixel_vector(img.height,image_pixel_row);

    for(int i= 0 ; i <img.step * img.height; ++i ){

        // int col = i / img.height;
        // int row = i % img.height;

        int col = i % img.step;
        int row = i / img.step;

        if (img.data[i] == white_pixel ){
            image_pixel_vector[row][col] = 1;
            find_white_flag = true;
        }
    }


    int max_col=0 ;
    int max_col_sum =0;
    for (int col =0 ; col < img.step; ++col){
        for (int row=0; row < img.height; ++row)
            image_pixel_row[col]+= image_pixel_vector[row][col];        
    
        if (image_pixel_row[col] > max_col_sum){
            max_col_sum = image_pixel_row[col];
            max_col = col;
        }
    }

    int position_step  =max_col;
    if (find_white_flag == false){
        // stop
         drive_robot(0.0, 0.0);
    }
     else{
        if (position_step < step_left_size){
        // left
        ROS_INFO("BALL is %3d of %3d: LEFT", max_col, img.step);
        drive_robot(0.02, -0.15);
        }

        else if (position_step > step_right_size){
        // right
        ROS_INFO("BALL is %3d of %3d: RIGHT", max_col, img.step);
        drive_robot(0.02, 0.15);
        }
    
       else{
        // straight
        ROS_INFO("BALL is %3d of %3d: FRONT", max_col, img.step);
         drive_robot(0.25, 0.0);
         }
     }
}

int main(int argc, char** argv)
{
    // Initialize the process_image node and create a handle to it
    ros::init(argc, argv, "process_image");
    ros::NodeHandle n;

    // Define a client service capable of requesting services from command_robot
    client = n.serviceClient<ball_chaser::DriveToTarget>("/ball_chaser/command_robot");

    // Subscribe to /camera/rgb/image_raw topic to read the image data inside the process_image_callback function
    ros::Subscriber sub1 = n.subscribe("/camera/rgb/image_raw", 10, process_image_callback);

    // Handle ROS communication events
    ros::spin();

    return 0;
}

```


```c++
// : drive_bot.cpp

#include "ros/ros.h"
#include "geometry_msgs/Twist.h"
//TODO: Include the ball_chaser "DriveToTarget" header file
#include "ball_chaser/DriveToTarget.h"

// ROS::Publisher motor commands;
ros::Publisher motor_command_publisher;

// TODO: Create a handle_drive_request callback function that executes whenever a drive_bot service is requested
// This function should publish the requested linear x and angular velocities to the robot wheel joints
// After publishing the requested velocities, a message feedback should be returned with the requested wheel velocities

bool handle_drive_request(ball_chaser::DriveToTarget::Request& req, ball_chaser::DriveToTarget::Response& res){

    ROS_INFO("handle_drive_request service is called - linear_x: %1.2f, angular_z: %1.2f", (float)req.linear_x, (float)req.angular_z);

    geometry_msgs::Twist motor_command;
    motor_command.linear.x = req.linear_x;
    motor_command.angular.z = req.angular_z;

    motor_command_publisher.publish(motor_command);
    
    // Wait 2 seconds
    ros::Duration(1).sleep();

    res.msg_feedback = "linear_x: " + std::to_string(motor_command.linear.x) + " , angular_z: " + std::to_string(motor_command.angular.z);
    ROS_INFO_STREAM(res.msg_feedback);

    return true;
}

int main(int argc, char** argv)
{
    // Initialize a ROS node
    ros::init(argc, argv, "drive_bot");

    // Create a ROS NodeHandle object
    ros::NodeHandle n;

    // Inform ROS master that we will be publishing a message of type geometry_msgs::Twist on the robot actuation topic with a publishing queue size of 10
    motor_command_publisher = n.advertise<geometry_msgs::Twist>("/cmd_vel", 10);

    // TODO: Define a drive /ball_chaser/command_robot service with a handle_drive_request callback function
    ros::ServiceServer service = n.advertiseService("/ball_chaser/command_robot", handle_drive_request);
    ROS_INFO("Ready to send motor velocity");
   
    // TODO: Handle ROS communication events
    ros::spin();

    return 0;
}
```




### 3. RESULTS  
![image](/assets/images/GoChaseIt1.gif)  
![image](/assets/images/GoChaseIt2.gif)  
![image](/assets/images/GoChaseIt3.gif)  

### 끝
