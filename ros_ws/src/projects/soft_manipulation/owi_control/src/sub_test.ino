/*
 * rosserial PubSub Example
 * Prints "hello world!" and toggles led
 */

#include <ros.h>
#include <std_msgs/String.h>
#include <std_msgs/Empty.h>
#include <std_msgs/Float64MultiArray.h>

#define in1 3
#define in2 4
#define enA 5

#define in3 7
#define in4 8
#define enB 6

ros::NodeHandle  nh;

void pwmCb(std_msgs::Float64MultiArray& msg){

  int u = (msg.data[0])*255;
  int v = (msg.data[1])*255;

//   Setting Direction
  if(u>=0){
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
  }
  else{
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
  }

// Setting PWM with Impulse
  if(v>=0){
    digitalWrite(in3, HIGH);
    digitalWrite(in4, LOW);
  }
  else{
    digitalWrite(in3, LOW);
    digitalWrite(in4, HIGH);
  }
  if(u < abs(78) && u!=0){
    analogWrite(enA,255);
    delay(2);
    analogWrite(enA,abs(u));
  }
  else{
    analogWrite(enA,abs(u));
  }
  if(v < abs(78) && v!=0){
    analogWrite(enB,255);
    delay(2);
    analogWrite(enB,abs(v));
  }
  else{
    analogWrite(enB,abs(v));
  }
//    Setting pwm
//  analogWrite(enA,abs(u));
//  analogWrite(enB,abs(v));
  
}

ros::Subscriber<std_msgs::Float64MultiArray> sub1("pwm",pwmCb);

void setup()
{
  pinMode(13,OUTPUT);
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(enA, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);
  pinMode(enB, OUTPUT);
  
  nh.initNode();
  nh.subscribe(sub1);
}

void loop()
{
  nh.spinOnce();
  //delay(500);
}
