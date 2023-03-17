 #include <ros/ros.h>
 #include <dynamixel_sdk/dynamixel_sdk.h>
 #include "assist_feeding/gripper_close.h"
 #include "assist_feeding/gripper_open.h"
 #include "classes/actuator.h"
 #include "classes/hand.h"

int main(){
  Hand h;
  h.grasp();
}