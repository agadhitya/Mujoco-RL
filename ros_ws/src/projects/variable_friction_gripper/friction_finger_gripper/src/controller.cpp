#include <friction_finger_gripper/finger.hpp>

class finger_controller
{
 	private:
		//Creating a pointer for fingers as the number of fingers are dynamic
		finger *fptr;
		DynamixelNode *nptr;
 	public:
 		finger_controller(int num_fingers,float parameters[]);
};

//Initializing the fingers and its parameters [ID,length,width,friction_coefficient]
finger_controller::finger_controller(int num_fingers, float parameters[])
{ 
 	std::vector<int> n;
	//Note:Need to add a condition for checking whether the input has correct number of parameters
 	for (int i =0;i<num_fingers;i++)
 	{
   		n.push_back(parameters[(4*i)]);
   		fptr= new finger(parameters[(4*i)+1],parameters[(4*i)+2],parameters[(4*i)+3]);
  	}
  	DynamixelNode D("XM",n);
  	ros::spin ();
   
}  

int main(int argc, char **argv)
{
 	ros::init(argc, argv, "finger_controller");
//{Motor_left_Id,Left_finger_length,left_finger_width,Left_finger_friction_coefficient,Motor_right_Id,Right_finger_length,Right_finger_width,Right_finger_friction_coefficient}
 	float parameters[]={21,2,3,4,20,6,7,8};   //Need to fix( Load params from yaml)
 	finger_controller h1(2,parameters);
 	return 0;
}

