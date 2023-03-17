#include "ros/ros.h"

#include <QApplication>

#include "mainWindow.hpp"

// main file
int main(int argc, char** argv) {
  ros::init(argc, argv, "ref_benchmark_main");
  ros::NodeHandle nh;

  QApplication app(argc, argv);
  
  MainWindow window{nh};
  window.show();
  
  return app.exec();
}
