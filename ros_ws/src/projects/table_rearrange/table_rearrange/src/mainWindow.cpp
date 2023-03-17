#include "mainWindow.hpp"

#include <iostream>

#include <QString>


MainWindow::MainWindow(ros::NodeHandle& nh) :
  ui(new Ui::MainWindow), n{nh}
{
    ui->setupUi(this);

    // Setup services
    setVelocityClient = n.serviceClient<moveit_planner::SetVelocity>("set_velocity_scaling");
    moveToPoseClient = n.serviceClient<moveit_planner::MovePose>("move_to_pose");
    moveToJointClient = n.serviceClient<moveit_planner::MoveJoint>("move_to_joint_space");
    moveAwayClient = n.serviceClient<moveit_planner::MoveAway>("move_away_point");
    moveCartClient = n.serviceClient<moveit_planner::MoveCart>("cartesian_move");
    randomizeClient = n.serviceClient<randomizer::Rand>("randomize");
    getImageClient = n.serviceClient<pcl_recorder::GetPointCloud>("get_point_cloud");
    getPoseClient = n.serviceClient<pose_estimator::PoseEstimation>("pose_estimator/gazebo_pose");
    gripperClient = n.serviceClient<franka_gripper_gazebo::GripMsg>("gazebo_franka_grip");

    // Setup UI connections
    connect(ui->pushButton, SIGNAL (released()), this, SLOT (randomizeClicked()));
    connect(ui->pushButton_2, SIGNAL (released()), this, SLOT (moveToStartClicked()));
    connect(ui->pushButton_3, SIGNAL (released()), this, SLOT (getReferenceClicked()));
    connect(ui->pushButton_4, SIGNAL (released()), this, SLOT (graspClicked()));
    connect(ui->pushButton_5, SIGNAL (released()), this, SLOT (getRandomizedClicked()));
    connect(ui->pushButton_6, SIGNAL (released()), this, SLOT (placeClicked()));
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::moveToStartClicked() {
  // Set velocity to be fast
  setVelocity(1.0);

  // Create message and send
  moveit_planner::MoveJoint msg;
  msg.request.val.push_back(0);		// 1
  msg.request.val.push_back(0);		// 2
  msg.request.val.push_back(0);		// 3
  msg.request.val.push_back(-3.1415/2);	// 4
  msg.request.val.push_back(0);		// 5
  msg.request.val.push_back(3.1415/2);	// 6
  msg.request.val.push_back(3.1415/4);	// 7
  msg.request.execute = true;
  if(!moveToJointClient.call(msg)) {
    ROS_ERROR("Could not move to start!");
  }
  else
    ROS_INFO("Moved to start!");
}

void MainWindow::randomizeClicked() {
  randomizer::Rand randMsg;
  randMsg.request.toStart = false;

  randomizeClient.call(randMsg);
}

void MainWindow::getReferenceClicked() {
  // Obtain reference pose
  getPoseClient.call(ref_pose_est);
}

void MainWindow::getRandomizedClicked() {
  getPoseClient.call(ran_pose_est);
  
  // Clear out all previous items
  for(int i = ui->comboBox->count()-1; i >= 0; --i)
    ui->comboBox->removeItem(i);

  // Loop over both responses and add the objects that are common
  for(int i = 0; i < ran_pose_est.response.detected_object_names.size(); ++i)
    if(getByName(ref_pose_est, ran_pose_est.response.detected_object_names[i]) != -1)
      ui->comboBox->insertItem(i, QString::fromStdString(ran_pose_est.response.detected_object_names[i]));
        
}

void MainWindow::graspClicked() {
  // Get the currently selected object in the comboBox
  int toGrasp = ui->comboBox->currentIndex();
  if(toGrasp == -1)
    return;
  // Set velocity to be fast
  setVelocity(1.0);

  // Store object
  graspedObject = ran_pose_est.response.detected_object_names[toGrasp];
  
  // For now, just try to grasp the first object in the ran_pose_est
  // Hardcoded grasp pose
  Eigen::MatrixXd graspPosition(4, 4);
  graspPosition = Eigen::MatrixXd::Zero(4, 4);
  // Rotation
  graspPosition(0,1) = -1;	// x-axis
  graspPosition(0,0) = 0;	// x-axis
  graspPosition(1,0) = -1;	// y-axis
  graspPosition(1,1) = 0;	// y-axis
  graspPosition(2,2) = -1;	// z-axis
  // Position
  graspPosition(0,3) = -0.0587;
  graspPosition(1,3) = 0;
  graspPosition(2,3) = 0.06069;
  graspPosition(3,3) = 1;

  // Convert from quat + position to transform matrix
  Eigen::MatrixXd objectPose = poseToTrans(ran_pose_est.response.detected_object_poses[toGrasp]);

  // Multiply to get final transform matrix
  Eigen::MatrixXd finalGraspTransform = objectPose * graspPosition;

  // Convert to quat
  Eigen::Matrix3d rot = finalGraspTransform.topLeftCorner(3, 3);
  Eigen::Quaterniond finalGraspQuat{rot};
  
  // Send final transform to server
  double graspDist = 0.3;
  geometry_msgs::Pose graspPose;
  graspPose.orientation.x = finalGraspQuat.x();
  graspPose.orientation.y = finalGraspQuat.y();
  graspPose.orientation.z = finalGraspQuat.z();
  graspPose.orientation.w = finalGraspQuat.w();
  graspPose.position.x = finalGraspTransform(0,3);
  graspPose.position.y = finalGraspTransform(1,3);
  graspPose.position.z = finalGraspTransform(2,3);

  // Send request
  moveit_planner::MoveAway msg;
  msg.request.pose = graspPose;
  msg.request.distance = graspDist;
  msg.request.execute = true;
  moveAwayClient.call(msg);

  // Open gripper
  franka_gripper_gazebo::GripMsg gripMsg;
  gripMsg.request.force = 100;
  gripperClient.call(gripMsg);

  // Slow down for approach
  setVelocity(0.1);

  // Move closer
  moveit_planner::MoveCart approachMsg;
  approachMsg.request.val.push_back(graspPose);
  approachMsg.request.execute = true;
  moveCartClient.call(approachMsg);

  // Grip
  gripMsg.request.force = -200;
  gripperClient.call(gripMsg);

  // Move away
  moveit_planner::MoveCart retreatMsg;
  retreatMsg.request.val.push_back(msg.response.awayPose);
  retreatMsg.request.execute = true;
  moveCartClient.call(retreatMsg);
}
void MainWindow::placeClicked() {
  // Make sure an object is grasped first
  if(graspedObject == "")
    return;

  // Obtain reference pose
  Eigen::MatrixXd refPose =
    poseToTrans(ref_pose_est.response.detected_object_poses[getByName(ref_pose_est,
								      graspedObject)]);

  // Move pose to ref pose
  // This is the previous hardcoded grasp pose
  Eigen::MatrixXd graspPosition(4, 4);
  graspPosition = Eigen::MatrixXd::Zero(4, 4);
  // Rotation
  graspPosition(0,1) = -1;	// x-axis
  graspPosition(0,0) = 0;	// x-axis
  graspPosition(1,0) = -1;	// y-axis
  graspPosition(1,1) = 0;	// y-axis
  graspPosition(2,2) = -1;	// z-axis
  // Position
  graspPosition(0,3) = -0.0587;
  graspPosition(1,3) = 0;
  graspPosition(2,3) = 0.06069;
  graspPosition(3,3) = 1;

  // Reference target
  Eigen::MatrixXd refGraspPose = refPose * graspPosition;

  Eigen::Matrix3d rot = refGraspPose.topLeftCorner(3, 3);
  Eigen::Quaterniond refGraspQuat{rot};
  
  // Send final transform to server
  double graspDist = 0.3;
  geometry_msgs::Pose graspPose;
  graspPose.orientation.x = refGraspQuat.x();
  graspPose.orientation.y = refGraspQuat.y();
  graspPose.orientation.z = refGraspQuat.z();
  graspPose.orientation.w = refGraspQuat.w();
  graspPose.position.x = refGraspPose(0,3);
  graspPose.position.y = refGraspPose(1,3);
  graspPose.position.z = refGraspPose(2,3);

  // Set velocity to slow
  setVelocity(0.1);

  // Move above target
  moveit_planner::MoveAway msg;
  msg.request.pose = graspPose;
  msg.request.distance = graspDist;
  msg.request.execute = true;
  moveAwayClient.call(msg);

  // Move to target
  moveit_planner::MoveCart approachMsg;
  approachMsg.request.val.push_back(graspPose);
  approachMsg.request.execute = true;
  moveCartClient.call(approachMsg);

  // Release
  franka_gripper_gazebo::GripMsg gripMsg;
  gripMsg.request.force = 100;
  gripperClient.call(gripMsg);
}


// Helper functions
void MainWindow::setVelocity(double val) {
  velCmd.request.velScaling = val;
  setVelocityClient.call(velCmd);
}
int MainWindow::getByName(const pose_estimator::PoseEstimation& poses, const std::string& name) {
  for(int i = 0; i < poses.response.detected_object_names.size(); ++i)
    if(name == poses.response.detected_object_names[i])
      return i;
  return -1;
}
Eigen::MatrixXd MainWindow::poseToTrans(const geometry_msgs::Pose& pose) {
  Eigen::MatrixXd retTrans(4, 4);
  retTrans = Eigen::MatrixXd::Zero(4, 4);
  Eigen::Quaterniond retQuat;
  retQuat.x() = pose.orientation.x;
  retQuat.y() = pose.orientation.y;
  retQuat.z() = pose.orientation.z;
  retQuat.w() = pose.orientation.w;
  retTrans.topLeftCorner(3, 3) = retQuat.normalized().toRotationMatrix();
  retTrans(0,3) = pose.position.x;
  retTrans(1,3) = pose.position.y;
  retTrans(2,3) = pose.position.z;
  retTrans(3,3) = 1;

  return retTrans;
}
