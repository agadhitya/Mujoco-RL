//This class records images and sensor readings in a rosbag
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <geometry_msgs/WrenchStamped.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/Point.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Int32.h>
#include<common_msgs_gl/Motor_position.h>
#include <memory>
#include <time.h>

class Recorder{
    private:
        // geometry_msgs::WrenchStamped msg;
        sensor_msgs::Image img;
        geometry_msgs::Point pnt;
        common_msgs_gl::Motor_position f;
        std_msgs::Bool b;
        std_msgs::Int32 a;
        rosbag::Bag* bag;
        std::string ft_sensor; std::string camera; std::string obj_pos;
        std::string encoder_val; std::string action;
        // ros::Subscriber sub1;
        ros::Subscriber sub2;
        ros::Subscriber sub3;
        ros::Subscriber sub4;
        ros::Subscriber sub5;
        ros::NodeHandle n;

        // void start_callback(const std_msgs::Bool begin){
        //     b = begin; 
        // }
        // void msg_callback(const geometry_msgs::WrenchStamped message){
        //     msg = message;
        // }
        void img_callback(const sensor_msgs::Image image){
            img = image;
        }
        void obj_pos_callback(const geometry_msgs::Point point_){
            pnt = point_;
        }
        void encoder_val_callback(const common_msgs_gl::Motor_position val){
            f = val;
        }
        void action_callback(const std_msgs::Int32 act){
            a = act;
        }
    public:
        Recorder(ros::NodeHandle& _n): n(_n){
            bag = new rosbag::Bag();
            bag->open("/home/gsathyanarayanan/test.bag", rosbag::bagmode::Write);    
            ROS_INFO("Opening bag");
            // n.getParam("topics/topic1", ft_sensor);
            n.getParam("topics/topic2", camera);
            n.getParam("topics/topic3", obj_pos);
            n.getParam("topics/topic4", encoder_val);
            n.getParam("topics/topic5", action);

            // sub1 = n.subscribe(ft_sensor, 1, &Recorder::msg_callback, this);
            sub2 = n.subscribe(camera, 1, &Recorder::img_callback, this);  
            sub3 = n.subscribe(obj_pos, 1, &Recorder::obj_pos_callback, this);
            sub4 = n.subscribe(encoder_val, 1, &Recorder::encoder_val_callback, this);
            sub5 = n.subscribe(action, 1, &Recorder::action_callback, this);

            ros::Rate r{30};
            while(ros::ok()){
                ros::Time t = ros::Time::now();
                // bag->write(ft_sensor, t, msg);
                bag->write(camera, t, img);
                bag->write(obj_pos, t, pnt);
                bag->write(encoder_val, t, f);
                bag->write(action, t, a);
                ros::spinOnce();
                r.sleep();
            }
        }

        ~Recorder(){
            bag->close();
            delete bag;
        }
};