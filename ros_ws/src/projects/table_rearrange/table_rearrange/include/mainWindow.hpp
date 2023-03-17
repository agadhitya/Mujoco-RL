#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "ros/ros.h"

#include "pose_estimator/PoseEstimation.h"
#include "sensor_msgs/PointCloud2.h"

#include "randomizer/Rand.h"
#include "geometry_msgs/Point.h"
#include "moveit_planner/MoveAway.h"
#include "moveit_planner/MovePose.h"
#include "moveit_planner/MoveCart.h"
#include "moveit_planner/MoveJoint.h"
#include "moveit_planner/SetVelocity.h"
#include "pcl_recorder/GetPointCloud.h"
#include "pose_estimator/PoseEstimation.h"
#include "franka_gripper_gazebo/GripMsg.h"

#include <QMainWindow>
#include "ui_mainWindow.h"

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace Ui {
  class MainWindow;
}

class MainWindow : public QMainWindow
{
  Q_OBJECT

public:
  explicit MainWindow(ros::NodeHandle &nh);
  virtual ~MainWindow();

private slots:
  // Callbacks
  void moveToStartClicked();
  void randomizeClicked();
  void getReferenceClicked();
  void getRandomizedClicked();
  void graspClicked();
  void placeClicked();

private:
  ros::NodeHandle& n;
  ros::ServiceClient setVelocityClient;
  ros::ServiceClient moveToPoseClient;
  ros::ServiceClient moveToJointClient;
  ros::ServiceClient moveAwayClient;
  ros::ServiceClient moveCartClient;
  ros::ServiceClient randomizeClient;
  ros::ServiceClient getImageClient;
  ros::ServiceClient getPoseClient;
  ros::ServiceClient gripperClient;
  
  Ui::MainWindow *ui;

  // Data
  sensor_msgs::PointCloud2 refPointCloud;
  pose_estimator::PoseEstimation ref_pose_est; // The estimated reference pose
  pose_estimator::PoseEstimation ran_pose_est; // The estimated randomized pose
  moveit_planner::SetVelocity velCmd;	       // A command to set the robot velocity
  std::string graspedObject;

  // Helper functions
  void setVelocity(double val);
  int getByName(const pose_estimator::PoseEstimation& poses, const std::string& name);
  Eigen::MatrixXd poseToTrans(const geometry_msgs::Pose& pose);
};

#endif // MAINWINDOW_H
