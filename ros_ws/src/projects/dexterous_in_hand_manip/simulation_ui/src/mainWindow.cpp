#include "mainWindow.hpp"
#include <simulation_ui/SetParam.h>
#include <gripper_controls/Holdcommand.h>
#include <gripper_controls/PositionCommand.h>
#include <std_srvs/Empty.h>
#include <iostream>

#include <QString>


MainWindow::MainWindow(ros::NodeHandle& nh) :
  ui(new Ui::MainWindow), n{nh}
{
    ui->setupUi(this);

    // Setup services:
    // Plugin communication
    setMuTorsionClient = n.serviceClient<simulation_ui::SetParam>("set_muTorsion");
    setUsePatchRadiusClient = n.serviceClient<simulation_ui::SetParam>("set_usePR");
    setPatchRadiusClient = n.serviceClient<simulation_ui::SetParam>("set_patchRadius");
    setSurfaceRadiusClient = n.serviceClient<simulation_ui::SetParam>("set_surfaceRadius");
    // Low level param communication
    setLowFrictionClient = n.serviceClient<simulation_ui::SetParam>("set_lowfriction");
    setHighFrictionClient = n.serviceClient<simulation_ui::SetParam>("set_highfriction");
    // High level controls communication
    HoldObjectClient = n.serviceClient<gripper_controls::Holdcommand>("Hold_object");
    RotateClockwiseClient = n.serviceClient<gripper_controls::PositionCommand>("Rotate_clockwise");
    RotateAnticlockwiseClient = n.serviceClient<gripper_controls::PositionCommand>("Rotate_anticlockwise");
    SlideLeftFingerDownClient = n.serviceClient<gripper_controls::PositionCommand>("Slide_Left_Finger_Down");
    SlideLeftFingerUpClient = n.serviceClient<gripper_controls::PositionCommand>("Slide_Left_Finger_Up");
    SlideRightFingerDownClient = n.serviceClient<gripper_controls::PositionCommand>("Slide_Right_Finger_Down");
    SlideRightFingerUpClient = n.serviceClient<gripper_controls::PositionCommand>("Slide_Right_Finger_Up");
    // Gazebo communication
    ResetWorldClient = n.serviceClient<std_srvs::Empty>("/gazebo/reset_world");
    ResetSimulationClient = n.serviceClient<std_srvs::Empty>("/gazebo/reset_simulation");
    //
    // // Setup UI connections
    // Plugin connections
    connect(ui->slider_muTorsion, SIGNAL (sliderMoved(int)), this, SLOT (computeMuTorsion(int)));
    connect(ui->spinbox_MuTorsion, SIGNAL (valueChanged(double)), this, SLOT (updateMuTorsionSlider(double)));
    connect(ui->spinbox_MuTorsion, SIGNAL (valueChanged(double)), this, SLOT (setMuTorsion(double)));
    connect(ui->checkBox_PatchRadius, SIGNAL (toggled(bool)), this, SLOT (setUsePatchRadius(bool)));
    connect(ui->slider_PatchRadius, SIGNAL (sliderMoved(int)), this, SLOT (computePatchRadius(int)));
    connect(ui->spinbox_PatchRadius, SIGNAL (valueChanged(double)), this, SLOT (updatePatchRadiusSlider(double)));
    connect(ui->spinbox_PatchRadius, SIGNAL (valueChanged(double)), this, SLOT (setPatchRadius(double)));
    connect(ui->slider_SurfaceRadius, SIGNAL (sliderMoved(int)), this, SLOT (computeSurfaceRadius(int)));
    connect(ui->spinbox_SurfaceRadius, SIGNAL (valueChanged(double)), this, SLOT (updateSurfaceRadiusSlider(double)));
    connect(ui->spinbox_SurfaceRadius, SIGNAL (valueChanged(double)), this, SLOT (setSurfaceRadius(double)));
    // Low level param connections
    connect(ui->slider_LowFriction, SIGNAL (sliderMoved(int)), this, SLOT (computeLowFriction(int)));
    connect(ui->spinbox_LowFriction, SIGNAL (valueChanged(double)), this, SLOT (updateLowFrictionSlider(double)));
    connect(ui->spinbox_LowFriction, SIGNAL (valueChanged(double)), this, SLOT (setLowFriction(double)));
    connect(ui->slider_HighFriction, SIGNAL (sliderMoved(int)), this, SLOT (computeHighFriction(int)));
    connect(ui->spinbox_HighFriction, SIGNAL (valueChanged(double)), this, SLOT (updateHighFrictionSlider(double)));
    connect(ui->spinbox_HighFriction, SIGNAL (valueChanged(double)), this, SLOT (setHighFriction(double)));
    // High level controls connections
    connect(ui->spinbox_HoldObjectLeft, SIGNAL (valueChanged(double)), this, SLOT (setHoldObjectLeft(double)));
    connect(ui->spinbox_HoldObjectRight, SIGNAL (valueChanged(double)), this, SLOT (setHoldObjectRight(double)));
    connect(ui->pushButton_HoldObject, SIGNAL (pressed()), this, SLOT (changeStatusMessage()));
    connect(ui->pushButton_HoldObject, SIGNAL (released()), this, SLOT (executeHoldObject()));
    connect(ui->spinbox_RotateClockwise, SIGNAL (valueChanged(double)), this, SLOT (setRotateClockwise(double)));
    connect(ui->pushButton_RotateClockwise, SIGNAL (pressed()), this, SLOT (changeStatusMessage()));
    connect(ui->pushButton_RotateClockwise, SIGNAL (released()), this, SLOT (executeRotateClockwise()));
    connect(ui->spinbox_RotateAnticlockwise, SIGNAL (valueChanged(double)), this, SLOT (setRotateAnticlockwise(double)));
    connect(ui->pushButton_RotateAnticlockwise, SIGNAL (pressed()), this, SLOT (changeStatusMessage()));
    connect(ui->pushButton_RotateAnticlockwise, SIGNAL (released()), this, SLOT (executeRotateAnticlockwise()));
    connect(ui->spinbox_SlideLeftFingerDown, SIGNAL (valueChanged(double)), this, SLOT (setSlideLeftFingerDown(double)));
    connect(ui->pushButton_SlideLeftFingerDown, SIGNAL (pressed()), this, SLOT (changeStatusMessage()));
    connect(ui->pushButton_SlideLeftFingerDown, SIGNAL (released()), this, SLOT (executeSlideLeftFingerDown()));
    connect(ui->spinbox_SlideLeftFingerUp, SIGNAL (valueChanged(double)), this, SLOT (setSlideLeftFingerUp(double)));
    connect(ui->pushButton_SlideLeftFingerUp, SIGNAL (pressed()), this, SLOT (changeStatusMessage()));
    connect(ui->pushButton_SlideLeftFingerUp, SIGNAL (released()), this, SLOT (executeSlideLeftFingerUp()));
    connect(ui->spinbox_SlideRightFingerDown, SIGNAL (valueChanged(double)), this, SLOT (setSlideRightFingerDown(double)));
    connect(ui->pushButton_SlideRightFingerDown, SIGNAL (pressed()), this, SLOT (changeStatusMessage()));
    connect(ui->pushButton_SlideRightFingerDown, SIGNAL (released()), this, SLOT (executeSlideRightFingerDown()));
    connect(ui->spinbox_SlideRightFingerUp, SIGNAL (valueChanged(double)), this, SLOT (setSlideRightFingerUp(double)));
    connect(ui->pushButton_SlideRightFingerUp, SIGNAL (pressed()), this, SLOT (changeStatusMessage()));
    connect(ui->pushButton_SlideRightFingerUp, SIGNAL (released()), this, SLOT (executeSlideRightFingerUp()));
    // Gazebo connections
    connect(ui->pushButton_ResetWorld, SIGNAL (clicked()), this, SLOT (resetWorld()));
    connect(ui->pushButton_ResetSimulation, SIGNAL (clicked()), this, SLOT (resetSimulation()));

}

MainWindow::~MainWindow()
{
    delete ui;
}

// Plugin
void MainWindow::computeMuTorsion(int val){
  double muTorsion{0.0};
  muTorsion = val/1000;
  ui->spinbox_MuTorsion->setValue(muTorsion);
}

void MainWindow::updateMuTorsionSlider(double val){
  int slider_val{0};
  slider_val = val*1000;
  ui->slider_muTorsion->setValue(slider_val);
}

void MainWindow::setMuTorsion(double muTorsion) {
  simulation_ui::SetParam srv;
  srv.request.param = muTorsion;
  setMuTorsionClient.call(srv);
}

void MainWindow::setUsePatchRadius(bool usePR){
  simulation_ui::SetParam srv;
  if (usePR)
    srv.request.param = 1;
  else
    srv.request.param = 0;
  setUsePatchRadiusClient.call(srv);
}

void MainWindow::computePatchRadius(int val){
  double patchRadius{0.0};
  patchRadius = val/1000;
  ui->spinbox_PatchRadius->setValue(patchRadius);
}

void MainWindow::updatePatchRadiusSlider(double val){
  int slider_val{0};
  slider_val = val*1000;
  ui->slider_PatchRadius->setValue(slider_val);
}

void MainWindow::setPatchRadius(double patchRadius) {
  simulation_ui::SetParam srv;
  srv.request.param = patchRadius;
  setPatchRadiusClient.call(srv);
}

void MainWindow::computeSurfaceRadius(int val){
  double surfaceRadius{0.0};
  surfaceRadius = val/1000;
  ui->spinbox_SurfaceRadius->setValue(surfaceRadius);
}

void MainWindow::updateSurfaceRadiusSlider(double val){
  int slider_val{0};
  slider_val = val*1000;
  ui->slider_SurfaceRadius->setValue(slider_val);
}

void MainWindow::setSurfaceRadius(double surfaceRadius) {
  simulation_ui::SetParam srv;
  srv.request.param = surfaceRadius;
  setSurfaceRadiusClient.call(srv);
}

// Low level param
void MainWindow::computeLowFriction(int val){
  double lowFriction{0.0};
  lowFriction = val/10000;
  ui->spinbox_LowFriction->setValue(lowFriction);
}

void MainWindow::updateLowFrictionSlider(double val){
  int slider_val{0};
  slider_val = val*10000;
  ui->slider_LowFriction->setValue(slider_val);
}

void MainWindow::setLowFriction(double lowFriction) {
  simulation_ui::SetParam srv;
  srv.request.param = lowFriction;
  setLowFrictionClient.call(srv);
}

void MainWindow::computeHighFriction(int val){
  double highFriction{0.0};
  highFriction = val/10000;
  ui->spinbox_HighFriction->setValue(highFriction);
}

void MainWindow::updateHighFrictionSlider(double val){
  int slider_val{0};
  slider_val = val*10000;
  ui->slider_HighFriction->setValue(slider_val);
}

void MainWindow::setHighFriction(double highFriction) {
  simulation_ui::SetParam srv;
  srv.request.param = highFriction;
  setHighFrictionClient.call(srv);
}
// High level controls
void MainWindow::setHoldObjectLeft(double left){
  HoldObjectLeftVal = left;
}

void MainWindow::setHoldObjectRight(double right){
  HoldObjectRightVal = right;
}

void MainWindow::executeHoldObject(){
  // StatusMessage("Executing Hold_object command");
  gripper_controls::Holdcommand srv_hold;
  srv_hold.request.left = HoldObjectLeftVal;
  srv_hold.request.right = HoldObjectRightVal;
  HoldObjectClient.call(srv_hold);
  StatusMessage("Waiting for command");
}

void MainWindow::setRotateClockwise(double data){
  RotateClockwiseVal = data;
}

void MainWindow::executeRotateClockwise(){
  // StatusMessage("Executing Slide_Left_Finger_Down command");
  gripper_controls::PositionCommand srv_rotate;
  srv_rotate.request.data = RotateClockwiseVal;
  RotateClockwiseClient.call(srv_rotate);
  StatusMessage("Waiting for command");
}

void MainWindow::setRotateAnticlockwise(double data){
  RotateAnticlockwiseVal = data;
}

void MainWindow::executeRotateAnticlockwise(){
  // StatusMessage("Executing Slide_Left_Finger_Down command");
  gripper_controls::PositionCommand srv_rotate;
  srv_rotate.request.data = RotateAnticlockwiseVal;
  RotateAnticlockwiseClient.call(srv_rotate);
  StatusMessage("Waiting for command");
}

void MainWindow::setSlideLeftFingerDown(double data){
  SlideLeftFingerDownVal = data;
}

void MainWindow::executeSlideLeftFingerDown(){
  // StatusMessage("Executing Slide_Left_Finger_Down command");
  gripper_controls::PositionCommand srv_slide;
  srv_slide.request.data = SlideLeftFingerDownVal;
  SlideLeftFingerDownClient.call(srv_slide);
  StatusMessage("Waiting for command");
}

void MainWindow::setSlideLeftFingerUp(double data){
  SlideLeftFingerUpVal = data;
}

void MainWindow::executeSlideLeftFingerUp(){
  // StatusMessage("Executing Slide_Left_Finger_Up command");
  gripper_controls::PositionCommand srv_slide;
  srv_slide.request.data = SlideLeftFingerUpVal;
  SlideLeftFingerUpClient.call(srv_slide);
  StatusMessage("Waiting for command");
}

void MainWindow::setSlideRightFingerDown(double data){
  SlideRightFingerDownVal = data;
}

void MainWindow::executeSlideRightFingerDown(){
  // StatusMessage("Executing Slide_Right_Finger_Down command");
  gripper_controls::PositionCommand srv_slide;
  srv_slide.request.data = SlideRightFingerDownVal;
  SlideRightFingerDownClient.call(srv_slide);
  StatusMessage("Waiting for command");
}

void MainWindow::setSlideRightFingerUp(double data){
  SlideRightFingerUpVal = data;
}

void MainWindow::executeSlideRightFingerUp(){
  // StatusMessage("Executing Slide_Right_Finger_Up command");
  gripper_controls::PositionCommand srv_slide;
  srv_slide.request.data = SlideRightFingerUpVal;
  SlideRightFingerUpClient.call(srv_slide);
  StatusMessage("Waiting for command");
}

// Gazebo
void MainWindow::resetWorld(){
  std_srvs::Empty srv_reset;
  ResetWorldClient.call(srv_reset);
}

void MainWindow::resetSimulation(){
  std_srvs::Empty srv_reset;
  ResetSimulationClient.call(srv_reset);
}

// Helper
void MainWindow::StatusMessage(QString message){
    // ui->textEdit->clear();
    // ui->textEdit->append(message);
    ui->textEdit->setText(message);
    ui->centralwidget->update();
    // ros::Duration d(1);
    // d.sleep();
    // ui->textEdit->update();
}

void MainWindow::changeStatusMessage(){
  StatusMessage("Executing command");
}
