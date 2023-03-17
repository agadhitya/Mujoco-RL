#include <ros.h>
#include <std_msgs/Bool.h>
#include <Servo.h>
#include <common_msgs_gl/SendBool.h>

//Create servo objects
Servo servo_right;
Servo servo_left;

ros::NodeHandle nh;

// The msg format which is used to communicate between server and the client
using common_msgs_gl::SendBool;


//Server callback function for changing the right finger friction surface to high or low
//1-low friction(Motor position=0)
//0-high fricton(motor position=140)
void callback_right(const SendBool::Request &req, SendBool:: Response &res)
{
  if(req.data)
 {
  servo_right.write(0);   // blink the led
  Serial.print("Friction Low");
 }
 else
 {
  Serial.print("Friction High");
  servo_right.write(140);
 }
}

//Server callback function for changing the left finger friction surface to high or low
//0-low friction(Motor position=140)
//1-high fricton(motor position=0)
void callback_left(const SendBool::Request &req, SendBool:: Response &res)
{
  if(req.data)
 {
  servo_left.write(140);   
  Serial.print("Friction Low");
 }
 else
 {
  Serial.print("Friction High");
  servo_left.write(0);
 }
}

// Create server for left and right friction surface
ros::ServiceServer<SendBool::Request, SendBool::Response> server_right("Friction_surface_Right",&callback_right);
ros::ServiceServer<SendBool::Request, SendBool::Response> server_left("Friction_surface_Left",&callback_left);

void setup()
{
  //Set the baud rate to 57600
  Serial.begin(57600);

  //Attach the left and right servos to the corresponding pin numbers
  servo_right.attach(10);
  servo_left.attach(9);

  nh.initNode();

  //Advertise the Service to ROS
  nh.advertiseService(server_right);
  nh.advertiseService(server_left);

}

void loop()
{
  nh.spinOnce();
  delay(1);
  }
