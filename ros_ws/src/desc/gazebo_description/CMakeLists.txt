cmake_minimum_required(VERSION 2.8.3)
project(gazebo_description)

find_package(catkin REQUIRED COMPONENTS xacro)

catkin_package(CATKIN_DEPENDS xacro)
