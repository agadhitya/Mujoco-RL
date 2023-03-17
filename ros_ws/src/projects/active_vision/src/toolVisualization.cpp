# include<active_vision/toolVisualization.h>

std::map<int, std::string> dirLookup = {{0, "Nil"},
                                        {1, "N"}, {2, "NE"},
																		    {3, "E"}, {4, "SE"},
																		    {5, "S"}, {6, "SW"},
																	 	    {7, "W"}, {8, "NW"}};

void setupViewer(ptCldVis::Ptr viewer, int nVps, std::vector<int> &vp){
  if(!(nVps==1 || nVps==2 || nVps==3 || nVps==4 || nVps==8 || nVps==9)) nVps = 1;
  viewer->initCameraParameters();
  vp.clear(); vp.resize(nVps);
  switch(nVps){
    case 1:
      viewer->createViewPort(0.0,0.0,1.0,1.0,vp[0]);
      break;
    case 2:
      viewer->createViewPort(0.0,0.0,0.5,1.0,vp[0]);
      viewer->createViewPort(0.5,0.0,1.0,1.0,vp[1]);
      break;
    case 3:
      viewer->createViewPort(0.0,0.5,0.5,1.0,vp[0]);
      viewer->createViewPort(0.5,0.5,1.0,1.0,vp[1]);
      viewer->createViewPort(0.25,0.0,0.75,0.5,vp[2]);
      break;
    case 4:
      viewer->createViewPort(0.0,0.5,0.5,1.0,vp[0]);
      viewer->createViewPort(0.5,0.5,1.0,1.0,vp[1]);
      viewer->createViewPort(0.0,0.0,0.5,0.5,vp[2]);
      viewer->createViewPort(0.5,0.0,1.0,0.5,vp[3]);
      break;
    case 8:
      viewer->createViewPort(0.00,0.66,0.33,1.00,vp[0]); viewer->createViewPortCamera(vp[0]);
      viewer->createViewPort(0.33,0.66,0.66,1.00,vp[1]); viewer->createViewPortCamera(vp[1]);
      viewer->createViewPort(0.66,0.66,1.00,1.00,vp[2]); viewer->createViewPortCamera(vp[2]);

      viewer->createViewPort(0.00,0.33,0.33,0.66,vp[3]); viewer->createViewPortCamera(vp[3]);
      viewer->createViewPort(0.33,0.33,0.66,0.66,vp[4]); viewer->createViewPortCamera(vp[4]);
      viewer->createViewPort(0.66,0.33,1.00,0.66,vp[5]); viewer->createViewPortCamera(vp[5]);

      viewer->createViewPort(0.00,0.00,0.50,0.33,vp[6]); viewer->createViewPortCamera(vp[6]);
      viewer->createViewPort(0.50,0.00,1.00,0.33,vp[7]); viewer->createViewPortCamera(vp[7]);
      break;
    case 9:
      viewer->createViewPort(0.00,0.00,0.33,0.33,vp[6]); viewer->createViewPortCamera(vp[6]);
      viewer->createViewPort(0.33,0.00,0.66,0.33,vp[5]); viewer->createViewPortCamera(vp[5]);
      viewer->createViewPort(0.66,0.00,1.00,0.33,vp[4]); viewer->createViewPortCamera(vp[4]);

      viewer->createViewPort(0.00,0.33,0.33,0.66,vp[7]); viewer->createViewPortCamera(vp[7]);
      viewer->createViewPort(0.33,0.33,0.66,0.66,vp[0]); viewer->createViewPortCamera(vp[0]);
      viewer->createViewPort(0.66,0.33,1.00,0.66,vp[3]); viewer->createViewPortCamera(vp[3]);

      viewer->createViewPort(0.00,0.66,0.33,1.00,vp[8]); viewer->createViewPortCamera(vp[8]);
      viewer->createViewPort(0.33,0.66,0.66,1.00,vp[1]); viewer->createViewPortCamera(vp[1]);
      viewer->createViewPort(0.66,0.66,1.00,1.00,vp[2]); viewer->createViewPortCamera(vp[2]);
      break;
  }
  for(int i = 0; i < nVps; i++){
    viewer->setBackgroundColor(0.5,0.5,0.5,vp[i]);
  }
  viewer->addCoordinateSystem(0.5);
}

// Fuction to add a rgb point cloud
void addRGB(ptCldVis::Ptr viewer, ptCldColor::ConstPtr cloud, std::string name,int vp){

  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB>(cloud,rgb,name,vp);
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,3,name,vp);
}

// Fuction to add a rgb point cloud with normals
void addRGBN(ptCldVis::Ptr viewer, ptCldColor::ConstPtr cloud, ptCldNormal::ConstPtr normal, std::string name,int vp){

  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB>(cloud,rgb,name,vp);
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,3,name,vp);
  viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(cloud,normal,1,0.01,name+"_normal",vp);
}

void setCamView(ptCldVis::Ptr viewer, std::vector<double> pose, pcl::PointXYZ &centre){
	pose[0]*=1.5;
	pcl::PointXYZ temp;
  temp.x = centre.x+pose[0]*sin(pose[2])*cos(pose[1]);
  temp.y = centre.y+pose[0]*sin(pose[2])*sin(pose[1]);
  temp.z = centre.z+pose[0]*cos(pose[2]);
  if(pose[2] == 0){
    viewer->setCameraPosition(temp.x,temp.y,temp.z,centre.x,centre.y,centre.z,cos(pose[1]+M_PI),sin(pose[1]),0);
  }else{
    viewer->setCameraPosition(temp.x,temp.y,temp.z,centre.x,centre.y,centre.z,0,0,1);
  }
}

void setCamView(ptCldVis::Ptr viewer, std::vector<double> pose, pcl::PointXYZ &centre, int vp){
	pose[0]*=1.5;
	pcl::PointXYZ temp;
  temp.x = centre.x+pose[0]*sin(pose[2])*cos(pose[1]);
  temp.y = centre.y+pose[0]*sin(pose[2])*sin(pose[1]);
  temp.z = centre.z+pose[0]*cos(pose[2]);
  if(pose[2] == 0){
    viewer->setCameraPosition(temp.x,temp.y,temp.z,centre.x,centre.y,centre.z,cos(pose[1]+M_PI),sin(pose[1]),0,vp);
  }else{
    viewer->setCameraPosition(temp.x,temp.y,temp.z,centre.x,centre.y,centre.z,0,0,1,vp);
  }
}

void addViewsphere(ptCldVis::Ptr viewer, int vp, pcl::PointXYZ centre, double &rad, bool all){
  if(all == true){
    pcl::PointXYZ temp;
    int vCtr = 0;
    for(int azimuthalAngle = 0; azimuthalAngle < 360; azimuthalAngle+=10){
      for(int polarAngle = 5; polarAngle <= 90; polarAngle+=10){
        temp.x = centre.x+rad*sin(polarAngle*(M_PI/180.0))*cos(azimuthalAngle*(M_PI/180.0));
        temp.y = centre.y+rad*sin(polarAngle*(M_PI/180.0))*sin(azimuthalAngle*(M_PI/180.0));
        temp.z = centre.z+rad*cos(polarAngle*(M_PI/180.0));
        vCtr++;
        // viewer->addSphere(temp,0.02,0,0,0.75,"Sph_"+std::to_string(vCtr),vp);
      }
    }
  }
  viewer->addSphere(centre,rad,0,0,1,"Viewsphere"+std::to_string(vp),vp);
  viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME,"Viewsphere"+std::to_string(vp));
}

keyboardEvent::keyboardEvent(ptCldVis::Ptr viewer, int num){
  type = num;
  called = false;
  counter = 0;
  ok = true;
  skipTo = false;
  mode = 1;
  dir = 0;
  if(type == 1)
    viewer->registerKeyboardCallback(&keyboardEvent::keyboardEventOccurredA, *this);
  else if(type == 2)
    viewer->registerKeyboardCallback(&keyboardEvent::keyboardEventOccurredB, *this);
  viewer->registerPointPickingCallback(&keyboardEvent::pointPickingEventOccurred, *this);
}

void keyboardEvent::keyboardEventOccurredA(const pcl::visualization::KeyboardEvent &event, void*){
  if(event.keyUp()){
    // std::cout << event.getKeySym() << std::endl;
    counter = 0;
    if(event.getKeySym() == "Right"){        counter = 1;    called = true;}
    else if(event.getKeySym() == "Left"){    counter = -1;   called = true;}
    else if(event.getKeySym() == "period"){  counter = 10;   called = true;}
    else if(event.getKeySym() == "comma"){   counter = -10;  called = true;}
    else if(event.getKeySym() == "Escape"){  ok = false;     called = true;}
    else if(event.getKeySym() == "z"){       skipTo = true;  called = true;}
  }
}

void keyboardEvent::keyboardEventOccurredB(const pcl::visualization::KeyboardEvent &event, void*){
  if(event.keyUp()){
    // std::cout << event.getKeySym() << std::endl;
    dir = -1;
    if      (event.getKeySym() == "KP_0"){ dir = 0;  called = true;}
    else if (event.getKeySym() == "KP_1"){ dir = 1;  called = true;}
    else if (event.getKeySym() == "KP_2"){ dir = 2;  called = true;}
    else if (event.getKeySym() == "KP_3"){ dir = 3;  called = true;}
    else if (event.getKeySym() == "KP_4"){ dir = 4;  called = true;}
    else if (event.getKeySym() == "KP_5"){ dir = 5;  called = true;}
    else if (event.getKeySym() == "KP_6"){ dir = 6;  called = true;}
    else if (event.getKeySym() == "KP_7"){ dir = 7;  called = true;}
    else if (event.getKeySym() == "KP_8"){ dir = 8;  called = true;}
    else if (event.getKeySym() == "0"){    dir = 0;  called = true;}
    else if (event.getKeySym() == "1"){    dir = 1;  called = true;}
    else if (event.getKeySym() == "2"){    dir = 2;  called = true;}
    else if (event.getKeySym() == "3"){    dir = 3;  called = true;}
    else if (event.getKeySym() == "4"){    dir = 4;  called = true;}
    else if (event.getKeySym() == "6"){    dir = 5;  called = true;}
    else if (event.getKeySym() == "7"){    dir = 6;  called = true;}
    else if (event.getKeySym() == "8"){    dir = 7;  called = true;}
    else if (event.getKeySym() == "9"){    dir = 8;  called = true;}
    else if (event.getKeySym() == "Escape"){ ok = false;  called = true;}
    else if (event.getKeySym() == "space"){  mode++; called = true;}
    mode %= 2;
    if(mode == 0) mode = 2;
  }
}

void keyboardEvent::pointPickingEventOccurred(const pcl::visualization::PointPickingEvent& event, void*){
  float x, y, z;
  if (event.getPointIndex() == -1) return;
  event.getPoint(x, y, z);
  std::cout << "[INOF] Point coordinate ( " << x << ", " << y << ", " << z << ")" << std::endl;
}

void keyboardEvent::help(){
  if (type == 1) {
    std::cout << "***Key Bindings***" << std::endl;
    std::cout << "\u2192 , \u2190 : Move between data (-1,+1)" << std::endl;
    std::cout << "\',\' , \'.\' : Move between data (-10,+10)" << std::endl;
    std::cout << "z : Goto a data number" << std::endl;
    std::cout << "Esc : Exit" << std::endl;
  }
}
