#include <friction_finger_gripper/finger.hpp>
#include <array>
#include <std_msgs/Bool.h>
#include <common_msgs_gl/SendBool.h>
#include <common_msgs_gl/SendDoubleArray.h>

#include <friction_finger_gripper/PositionCommand.h>
#include <friction_finger_gripper/Holdcommand.h>
#include <friction_finger_gripper/ActionCommand.h>

class hand
{
    private:
    int finger_state;

    float read_position(int motor_num)
    {
        ros::NodeHandle n;
        ros::service::waitForService("read_pos");
        ros::ServiceClient client_read_pos = n.serviceClient<common_msgs_gl::GetDoubleArray>("read_pos");
        common_msgs_gl::GetDoubleArray srv;

        if (client_read_pos.call(srv))
        {
            ROS_INFO("Read_position_success");
            //ROS_INFO("%f",srv.response.data[0]);
            return srv.response.data[motor_num - 1];
        }
        else
        {
            ROS_INFO("Failure");
            return -1;
        }
    }

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

    bool set_friction_left(bool friction_surface)
    {
        ros::NodeHandle n;
        ros::service::waitForService("Friction_surface_Right");
        ros::ServiceClient client_torque = n.serviceClient<common_msgs_gl::SendBool>("Friction_surface_Right");
        common_msgs_gl::SendBool srv;
        srv.request.data = friction_surface;
        // if(client_torque.call(srv))

        if (1)
        {
            client_torque.call(srv);
            ros::Duration(0.5).sleep();
            ROS_INFO("Friction_surface_set_Success");
            return 1;
        }
        else
        {
            ROS_INFO("Failure");
            return 0;
        }
    }

    bool set_friction_right(bool friction_surface)
    {
        ros::NodeHandle n;
        ros::service::waitForService("Friction_surface_Left");
        ros::ServiceClient client_torque = n.serviceClient<common_msgs_gl::SendBool>("Friction_surface_Left");
        common_msgs_gl::SendBool srv;
        srv.request.data = friction_surface;
        //if(client_torque.call(srv))
        if (1)
        {
            client_torque.call(srv);
            ros::Duration(0.5).sleep();
            ROS_INFO("Friction_surface_set_Success");
            return 1;
        }
        else
        {
            ROS_INFO("Failure");
            return 0;
        }
    }

    public:
    hand()
    {
        std::cout<<"Hand Object created";
        finger_state = 0; // 1->Slide Left, 2-> Slide Right, 3-> Rotate
    }
    bool slide_left_down(friction_finger_gripper::PositionCommand::Request &req, friction_finger_gripper::PositionCommand::Response &res)
    {
        
        bool set_modes, set_friction_l, set_friction_r, send_torque, send_pos;
        // Set modes - Left -> Position, Right -> Torque (3 -> Position, 0 -> Torque)
        int modes[] = {3, 0};
        set_modes = set_actuator_modes(2, modes);
        
        if (finger_state != 1)
        {
            // Set Friction Surfaces - Left -> Low, Right -> High
            set_friction_l = set_friction_right(false);
            set_friction_r = set_friction_left(true);
            finger_state = 1;
        }
        send_torque = command_torque(1, 0.15);

        if (send_torque)
        {
            float initial_position;
            initial_position = read_position(1);
            for(int i=initial_position; i>=req.data; i=i-0.01)
            {
                ros::Duration(0.5).sleep();
                send_pos = command_position(0, i);
            }
            if (send_pos)
                return 1;
            else
            {
                ROS_ERROR("Sending Position Values Failed");
                return 0;
            }
        }
        else
        {
            ROS_ERROR("Setting Torque Values Failed");
            return 0;
        }
    }

    bool slide_left_up(friction_finger_gripper::PositionCommand::Request &req, friction_finger_gripper::PositionCommand::Response &res)
    {
        
        bool set_modes, set_friction_l, set_friction_r, send_torque, send_pos;
        // Set modes - Left -> Torque, Right -> Position (3 -> Position, 0 -> Torque)
        int modes[] = {0, 3};
        set_modes = set_actuator_modes(2, modes);

        if (finger_state != 1)
        {
            // Set Friction Surfaces - Left -> Low, Right -> High
            set_friction_l = set_friction_right(false);
            set_friction_r = set_friction_left(true);
            finger_state = 1;
        }

        send_torque = command_torque(0, 0.15);

        if (send_torque)
        {
            float initial_position;
            initial_position = read_position(2);
            for(int i=initial_position; i>=req.data; i=i-0.01)
            {
                ros::Duration(0.5).sleep();
                send_pos = command_position(1, i);
            }

            if (send_pos)
                return 1;

            else
            {
                ROS_ERROR("Sending Position values failed");
                return 0;
            }
        }
        else
        {
            ROS_ERROR("Sending Torque values failed");
            return 0;
        }
    }

    bool slide_right_down(friction_finger_gripper::PositionCommand::Request &req, friction_finger_gripper::PositionCommand::Response &res)
    {
        
        bool set_modes, set_friction_l, set_friction_r, send_torque, send_pos;
        // Set modes - Left -> Position, Right -> Torque (3 -> Position, 0 -> Torque)
        int modes[] = {0, 3};
        set_modes = set_actuator_modes(2, modes);
        if (finger_state != 2)
        {
            // Set Friction Surfaces - Left -> High, Right -> Low
            set_friction_l = set_friction_left(true);
            set_friction_r = set_friction_right(false);
            finger_state = 2;
        }
        

        send_torque = command_torque(0, 0.15);

        if (send_torque)
        {
            float initial_position;
            initial_position = read_position(2);

            for(int i=initial_position; i>=req.data; i=i-0.01)
            {
                ros::Duration(0.5).sleep();
                send_pos = command_position(1, i);
            }

            if (send_pos)
            {
                return 1;
            }
            else
            {
                ROS_ERROR("Sending Position Values Failed");
                return 0;
            }
        }
        else
        {
            ROS_ERROR("Sending Torque Values failed");
            return 0;
        }
    }

    bool slide_right_up(friction_finger_gripper::PositionCommand::Request &req, friction_finger_gripper::PositionCommand::Response &res)
    {
        bool set_modes, set_friction_l, set_friction_r, send_torque, send_pos;
        // Set modes - Left -> Position, Right -> Torque (3 -> Position, 0-> Torque)
        int modes[] = {3, 0};
        set_modes = set_actuator_modes(2, modes);

        if (finger_state != 2)
        {
            // Set Friction Surfaces - Left -> High, Right -> Low
            set_friction_l = set_friction_left(true);
            set_friction_r = set_friction_right(false);
            finger_state = 2;
        }
        
        send_torque = command_torque(1, 0.15);
        if (send_torque)
        {
            float initial_position;
            initial_position = read_position(1);
            for(int i=initial_position; i>=req.data; i=i-0.01)
            {
                ros::Duration(0.5).sleep();
                send_pos = command_position(0, i);
            }
            if (send_pos)
                return 1;
            else
            {
                ROS_ERROR("Sending Position Values Failed");
                return 0;
            }
        }
        else
        {
            ROS_ERROR("Setting Torque Values Failed");
            return 0;
        }
    }

    bool rotate_anticlockwise(friction_finger_gripper::PositionCommand::Request &req, friction_finger_gripper::PositionCommand::Response &res)
    {
        bool set_modes, set_friction_l, set_friction_r, send_torque, send_pos;
        // Set modes Left -> Torque, Right -> Position (3 -> Position, 0 -> Torque)
        int modes[] = {0, 3};
        set_modes = set_actuator_modes(2, modes);

        if (finger_state != 3)
        {
            // Setting Friction Surfaces Left -> High, Right -> High
            set_friction_l = set_friction_left(true);
            set_friction_r = set_friction_right(true);
            finger_state = 3;
        }
        
        send_torque = command_torque(0, 0.15);

        if (send_torque)
        {
            float initial_position;
            initial_position = read_position(2);

            for (float i = initial_position; i >= req.data; i = i - 0.01)
                send_pos = command_position(1, i);

            if (send_pos)
            {
                ROS_INFO("SS");
                return 1;
            }
            else
                return 0;
        }
        else
            return 0;
    }

    bool rotate_clockwise(friction_finger_gripper::PositionCommand::Request &req, friction_finger_gripper::PositionCommand::Response &res)
    {

        bool set_modes, set_friction_l, set_friction_r, send_torque, send_pos;
        // Set modes Left -> Position, Right -> Torque (3 -> Position, 0 -> Torque)
        int modes[] = {3, 0};
        set_modes = set_actuator_modes(2, modes);

        if (finger_state != 3)
        {
            // Setting Friction Surfaces Left -> High, Right -> High
            set_friction_l = set_friction_left(true);
            set_friction_r = set_friction_right(true);
            finger_state = 3;
        }

        send_torque = command_torque(1, 0.15);
        if (send_torque)
        {
            float initial_position;
            initial_position = read_position(1);

            for (float i = initial_position; i >= req.data; i = i - 0.01)
                send_pos = command_position(0, i);

            if (send_pos)
            {
                ROS_INFO("SS");
                return 1;
            }
            else
                return 0;
        }
        else
            return 0;
    }

    bool hold_object1(friction_finger_gripper::Holdcommand::Request &req, friction_finger_gripper::Holdcommand::Response &res)
    {
        bool set_modes, send_pos1, send_pos2, set_friction_l, set_friction_r;
        int modes1[] = {3, 3};
        set_modes = set_actuator_modes(2, modes1);

        send_pos1 = command_position(0, req.left);
        send_pos2 = command_position(1, req.right);

        if (finger_state != 3)
        {
            // Setting Friction Surfaces Left -> High, Right -> High
            set_friction_l = set_friction_left(true);
            set_friction_r = set_friction_right(true);
            finger_state = 3;
        }

        ros::Duration(1).sleep();

        if (send_pos1 && send_pos2)
            return true;
        else
            return false;
    }
};


int main(int argc, char **argv)
{
    ros::init(argc, argv, "High_level_action_server");
    ros::NodeHandle n;
    hand h;
    ros::ServiceServer service_1 = n.advertiseService("Slide_Left_Finger_Down", &hand::slide_left_down, &h);
    ros::ServiceServer service_2 = n.advertiseService("Slide_Left_Finger_Up", &hand::slide_left_up, &h);
    ros::ServiceServer service_3 = n.advertiseService("Slide_Right_Finger_Down", &hand::slide_right_down, &h);
    ros::ServiceServer service_4 = n.advertiseService("Slide_Right_Finger_Up", &hand::slide_right_up, &h);
    ros::ServiceServer service_5 = n.advertiseService("Rotate_anticlockwise", &hand::rotate_anticlockwise, &h);
    ros::ServiceServer service_6 = n.advertiseService("Rotate_clockwise", &hand::rotate_clockwise, &h);
    ros::ServiceServer service_7 = n.advertiseService("Hold_object", &hand::hold_object1, &h);

    ros::spin();
    return 0;
}
