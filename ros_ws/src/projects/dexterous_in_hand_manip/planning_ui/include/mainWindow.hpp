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
  void executeResetSequence();
  void executePrepareGrasp();
  void displayStartMessage();
  void executePlan();
  void set_dVal(double);
  void set_zVal(double);
  void executeAdjustArm();
  void executeReleaseObject();
  void resetWorld();
  void resetSimulation();
  void resetArm();

private:
  ros::NodeHandle& n;
  ros::Duration dur{3};
  ros::ServiceClient ResetSequenceClient;
  ros::ServiceClient PrepareGraspClient;
  ros::ServiceClient PlanExecuterClient;
  ros::ServiceClient AdjustArmClient;
  ros::ServiceClient ReleaseObjectClient;
  ros::ServiceClient ResetWorldClient;
  ros::ServiceClient ResetSimulationClient;
  ros::ServiceClient ResetArmClient;

  Ui::MainWindow *ui;
  double dVal{8.0};
  double zVal{5.0};
};

#endif // MAINWINDOW_H
