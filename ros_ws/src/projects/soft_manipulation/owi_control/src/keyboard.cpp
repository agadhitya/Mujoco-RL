#include "ros/ros.h"
#include <termios.h>

#include "std_msgs/Float64MultiArray.h"
#include "std_msgs/Float64.h"

char getch(){
    fd_set set;
	struct timeval timeout;
	int rv;
	char buff = 0;
	int len = 1;
	int filedesc = 0;
	FD_ZERO(&set);
	FD_SET(filedesc, &set);
	
	timeout.tv_sec = 0;
	timeout.tv_usec = 1000;

	rv = select(filedesc + 1, &set, NULL, NULL, &timeout);

	struct termios old = {0};
	if (tcgetattr(filedesc, &old) < 0)
		ROS_ERROR("tcsetattr()");
	old.c_lflag &= ~ICANON;
	old.c_lflag &= ~ECHO;
	old.c_cc[VMIN] = 1;
	old.c_cc[VTIME] = 0;
	if (tcsetattr(filedesc, TCSANOW, &old) < 0)
		ROS_ERROR("tcsetattr ICANON");

	if(rv == -1)
		ROS_ERROR("select");
	else if(rv == 0)
		ROS_INFO("no_key_pressed");
	else
		read(filedesc, &buff, len );

	old.c_lflag |= ICANON;
	old.c_lflag |= ECHO;
	if (tcsetattr(filedesc, TCSADRAIN, &old) < 0)
		ROS_ERROR ("tcsetattr ~ICANON");
	return (buff);
}
int main(int argc, char **argv){

    ros::init(argc, argv,"keyboard_node");
    ros::NodeHandle n;
    ros::Publisher key_pub = n.advertise<std_msgs::Float64MultiArray>("pwm",1);

    ros::Rate loop_rate(30);
    while(ros::ok()){
        int c = getch();
        std_msgs::Float64MultiArray vel_msg;    //Velocity Message for Pub
        vel_msg.data.resize(2);

        if (c == 'w'){
            ROS_INFO("%c", c);
    
            vel_msg.data[0] = 0.3;
            vel_msg.data[1] = 0.0;

            key_pub.publish(vel_msg);
        }
        else if (c == 's'){
            ROS_INFO("%c", c);
    
            vel_msg.data[0] = -0.3;
            vel_msg.data[1] = 0.0;

            key_pub.publish(vel_msg);
        }
        else if (c == 'a'){
            ROS_INFO("%c", c);
    
            vel_msg.data[0] = 0.0;
            vel_msg.data[1] = -0.3;

            key_pub.publish(vel_msg);
        }
        else if (c == 'd'){
            ROS_INFO("%c", c);
    
            vel_msg.data[0] = 0.0;
            vel_msg.data[1] = 0.3;

            key_pub.publish(vel_msg);
        }
        else{
            
            vel_msg.data[0] = 0.0;
            vel_msg.data[1] = 0.0;

            key_pub.publish(vel_msg);
        }
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}