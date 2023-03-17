#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "ros/ros.h"

#include <QMainWindow>
#include "ui_mainWindow.h"

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
  void computeMuTorsion(int);
  void updateMuTorsionSlider(double);
  void setMuTorsion(double);
  void setUsePatchRadius(bool);
  void computePatchRadius(int);
  void updatePatchRadiusSlider(double);
  void setPatchRadius(double);
  void computeSurfaceRadius(int);
  void updateSurfaceRadiusSlider(double);
  void setSurfaceRadius(double);
  void computeLowFriction(int);
  void updateLowFrictionSlider(double);
  void setLowFriction(double);
  void computeHighFriction(int);
  void updateHighFrictionSlider(double);
  void setHighFriction(double);
  void setHoldObjectLeft(double);
  void setHoldObjectRight(double);
  void executeHoldObject();
  void setRotateClockwise(double);
  void executeRotateClockwise();
  void setRotateAnticlockwise(double);
  void executeRotateAnticlockwise();
  void setSlideLeftFingerDown(double);
  void executeSlideLeftFingerDown();
  void setSlideLeftFingerUp(double);
  void executeSlideLeftFingerUp();
  void setSlideRightFingerDown(double);
  void executeSlideRightFingerDown();
  void setSlideRightFingerUp(double);
  void executeSlideRightFingerUp();
  void resetWorld();
  void resetSimulation();
  void changeStatusMessage();

private:
  ros::NodeHandle& n;
  ros::ServiceClient setMuTorsionClient;
  ros::ServiceClient setUsePatchRadiusClient;
  ros::ServiceClient setPatchRadiusClient;
  ros::ServiceClient setSurfaceRadiusClient;
  ros::ServiceClient setLowFrictionClient;
  ros::ServiceClient setHighFrictionClient;
  ros::ServiceClient HoldObjectClient;
  ros::ServiceClient RotateClockwiseClient;
  ros::ServiceClient RotateAnticlockwiseClient;
  ros::ServiceClient SlideLeftFingerDownClient;
  ros::ServiceClient SlideLeftFingerUpClient;
  ros::ServiceClient SlideRightFingerDownClient;
  ros::ServiceClient SlideRightFingerUpClient;
  ros::ServiceClient ResetWorldClient;
  ros::ServiceClient ResetSimulationClient;

  Ui::MainWindow *ui;
  double HoldObjectLeftVal;
  double HoldObjectRightVal;
  double RotateClockwiseVal;
  double RotateAnticlockwiseVal;
  double SlideLeftFingerDownVal;
  double SlideLeftFingerUpVal;
  double SlideRightFingerDownVal;
  double SlideRightFingerUpVal;

  void StatusMessage(QString);
};

#endif // MAINWINDOW_H
