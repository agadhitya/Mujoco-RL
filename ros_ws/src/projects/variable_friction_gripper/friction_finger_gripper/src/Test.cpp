#include <friction_finger_gripper/finger.hpp>
#include <array>
#include <std_msgs/Bool.h>
#include <common_msgs_gl/SendBool.h>
#include <common_msgs_gl/SendDoubleArray.h>

#include <friction_finger_gripper/PositionCommand.h>
#include <friction_finger_gripper/Holdcommand.h>
#include <friction_finger_gripper/ActionCommand.h>

#include <ctime>

using namespace std;

bool command_position(int motor_num, float position)
{
    ros::NodeHandle n;
    ros::service::waitForService("cmd_pos_ind");
    ros::ServiceClient client_position = n.serviceClient<common_msgs_gl::SendDoubleArray>("cmd_pos_ind");
    common_msgs_gl::SendDoubleArray srv;
    srv.request.data = {motor_num, position};

    if (client_position.call(srv))
    {

        //XXX: sleep deactivated for speed testing
        //ros::Duration(0.5).sleep();
        ROS_INFO("Command_position[%d]_Success=[%f]", motor_num, position);
        return 1;
    }
    else
    {
        ROS_INFO("Failure");
        return -1;
    }
}

bool set_actuator_modes(int size, int modes[])
{
    ros::NodeHandle n;
    ros::service::waitForService("set_operating_mode");
    ros::ServiceClient client_operating_mode = n.serviceClient<common_msgs_gl::SendIntArray>("set_operating_mode");
    common_msgs_gl::SendIntArray srv;
    int values[size];

    for (int i = 0; i < size; i++)
    {
        values[i] = modes[i];
    }
    srv.request.data = {values[0], values[1]};
    if (client_operating_mode.call(srv))
    {
        ROS_INFO("Set_operating_mode_Success");
        return 1;
    }
    else
    {
        ROS_INFO("Failure");
        return -1;
    }
}
bool command_torque(int motor_num, float torque)
{
    ros::NodeHandle n;
    ros::service::waitForService("cmd_torque_ind");
    ros::ServiceClient client_torque = n.serviceClient<common_msgs_gl::SendDoubleArray>("cmd_torque_ind");
    common_msgs_gl::SendDoubleArray srv;
    srv.request.data = {motor_num, torque};

    if (client_torque.call(srv))
    {
        //XXX: sleep deactivated for speed testing
        //ros::Duration(0.5).sleep();
        ROS_INFO("Command_torque_Success");
        return 1;
    }
    else
    {
        ROS_INFO("Failure");
        return -1;
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "Testing_time");

    float i;
    bool send_pos, set_modes, send_torque;
    clock_t begin = clock();
    //code_to_time();
    
    int modes[] = {3, 3};
    set_modes = set_actuator_modes(2, modes);
    /*
    send_torque = command_torque(0, 0.15);

    for (i = 0.75; i > 0.65; i=i-0.01)
    {   

        send_pos = command_position(1, i);
    }

    clock_t end = clock();
    double elapsed_secs = double(double(end - begin)/CLOCKS_PER_SEC);
    
    cout<<"Time elapsed"<<elapsed_secs;
    */
    return 0;
}
