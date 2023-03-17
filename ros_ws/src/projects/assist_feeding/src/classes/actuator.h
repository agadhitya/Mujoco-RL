#ifndef ACTUATOR_CLASS
#define ACTUATOR_CLASS
#include <string>
#include <iostream>

using namespace std;

class Actuator{
    public:
    //members
    int ID;
    string type;

    bool torque_en();
    int read_pos();
    bool write_pos();


    //methods
};
#endif