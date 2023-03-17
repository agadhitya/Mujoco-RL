#include "mainWindow.hpp"
#include <manipulation_exp/Sequence.h>
#include <manipulation_planning/PlanExe.h>
#include <manipulation_planning/ArmAdjust.h>
#include <gripper_controls/Holdcommand.h>
#include <arm_controls/PoseChange.h>
#include <std_srvs/Empty.h>
#include <iostream>

#include <QString>


MainWindow::MainWindow(ros::NodeHandle& nh) :
  ui(new Ui::MainWindow), n{nh}
{
    ui->setupUi(this);

    // Setup services:
    // Initialization communication
    ResetSequenceClient = n.serviceClient<manipulation_exp::Sequence>("ih_manip/reset_sequence");
    PrepareGraspClient = n.serviceClient<manipulation_exp::Sequence>("ih_manip/prepare_grasp");
    PlanExecuterClient = n.serviceClient<manipulation_planning::PlanExe>("execute_plan");
    AdjustArmClient = n.serviceClient<manipulation_planning::ArmAdjust>("adjust_arm");
    ReleaseObjectClient = n.serviceClient<gripper_controls::Holdcommand>("Hold_object");
    // Gazebo communication
    ResetWorldClient = n.serviceClient<std_srvs::Empty>("/gazebo/reset_world");
    ResetSimulationClient = n.serviceClient<std_srvs::Empty>("/gazebo/reset_simulation");
    ResetArmClient = n.serviceClient<arm_controls::PoseChange>("reset_arm");
    //
    // // Setup UI connections
    // Initialization connections
    connect(ui->pushButton_resetSequence, SIGNAL (clicked()), this, SLOT (executeResetSequence()));
    connect(ui->pushButton_prepareGrasp, SIGNAL (clicked()), this, SLOT (executePrepareGrasp()));
    connect(ui->pushButton_executePlan, SIGNAL (pressed()), this, SLOT (displayStartMessage()));
    connect(ui->pushButton_executePlan, SIGNAL (released()), this, SLOT (executePlan()));
    connect(ui->doubleSpinBox_d, SIGNAL (valueChanged(double)), this, SLOT (set_dVal(double)));
    connect(ui->doubleSpinBox_z, SIGNAL (valueChanged(double)), this, SLOT (set_zVal(double)));
    connect(ui->pushButton_adjustArm, SIGNAL (clicked()), this, SLOT (executeAdjustArm()));
    connect(ui->pushButton_releaseObject, SIGNAL (clicked()), this, SLOT (executeReleaseObject()));

    // Gazebo connections
    connect(ui->pushButton_resetWorld, SIGNAL (clicked()), this, SLOT (resetWorld()));
    connect(ui->pushButton_resetSimulation, SIGNAL (clicked()), this, SLOT (resetSimulation()));
    connect(ui->pushButton_resetArm, SIGNAL (clicked()), this, SLOT (resetArm()));

}

MainWindow::~MainWindow()
{
    delete ui;
}

// Initialization
void MainWindow::executeResetSequence(){
  manipulation_exp::Sequence srv_resetseq;
  srv_resetseq.request.start = 1;
  ResetSequenceClient.call(srv_resetseq);
}

// Approach to a pre-grasp position
void MainWindow::executePrepareGrasp(){
  manipulation_exp::Sequence srv_prepgrasp;
  srv_prepgrasp.request.start = 1;
  PrepareGraspClient.call(srv_prepgrasp);
}

void MainWindow::displayStartMessage(){

  QString start;
  QString z_init = QString::number(zVal);
  QString d_init = QString::number(dVal);
  QString text_start;
  text_start = "Executing plan with \n";
  start = text_start + "d1 & d2: " + d_init + "\n" + "z: " + z_init;
  ui->textEdit->setText(start);
}

// Execute a manipulation plan and report results

void MainWindow::executePlan(){

  manipulation_planning::PlanExe srv_planexe;
  srv_planexe.request.execute = true;
  PlanExecuterClient.call(srv_planexe);
  float z{srv_planexe.response.z};
  float d{srv_planexe.response.d};
  QString z_final = QString::number(z);
  QString d_final = QString::number(d);
  QString results;
  QString text_results;
  text_results = "Results \n";
  results = text_results + "d1 & d2: " + d_final + "\n" + "z: " + z_final;
  ui->textEdit->setText(results);
}

void MainWindow::set_dVal(double val){
  dVal = val;
}

void MainWindow::set_zVal(double val){
  zVal = val;
}

// Adjust arm to achieve desired state
void MainWindow::executeAdjustArm(){
  manipulation_planning::ArmAdjust srv_armadjust;
  srv_armadjust.request.d = dVal;
  srv_armadjust.request.z = zVal;
  AdjustArmClient.call(srv_armadjust);
}

// Release the object
void MainWindow::executeReleaseObject(){
gripper_controls::Holdcommand srv_release;
srv_release.request.left = 0.5;
srv_release.request.right = 0.5;
ReleaseObjectClient.call(srv_release);
}

// Gazebo related operations
void MainWindow::resetWorld(){
  std_srvs::Empty srv_reset;
  ResetWorldClient.call(srv_reset);
}

void MainWindow::resetSimulation(){
  std_srvs::Empty srv_reset;
  ResetSimulationClient.call(srv_reset);
}

void MainWindow::resetArm(){
  arm_controls::PoseChange srv_reset;
  srv_reset.request.execute = true;
  ResetArmClient.call(srv_reset);
}

// Helper
