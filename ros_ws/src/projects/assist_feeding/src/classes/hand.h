#ifndef HAND_CLASS
#define HAND_CLASS

class Hand
{
public:
    //members
    //int n; // number of fingers
    //int ID[n]; // array of actuator IDs

#include "control_table.h"

    //methods
    bool grasp();
    bool drop();
    bool get_hand_state();
    int group_move();
    bool finger_move();

private:
    int getch();
    int kbhit(void);
    //members

    //methods
};
#endif