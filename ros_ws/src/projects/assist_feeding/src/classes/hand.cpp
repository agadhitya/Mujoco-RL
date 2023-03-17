#include "hand.h"
#include "actuator.h"

#ifdef __linux__
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#elif defined(_WIN32) || defined(_WIN64)
#include <conio.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include "dynamixel_sdk/dynamixel_sdk.h"

int Hand::getch()
{
#ifdef __linux__
    struct termios oldt, newt;
    int ch;
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    ch = getchar();
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    return ch;
#elif defined(_WIN32) || defined(_WIN64)
    return _getch();
#endif
}

int Hand::kbhit(void)
{
#ifdef __linux__
    struct termios oldt, newt;
    int ch;
    int oldf;

    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

    ch = getchar();

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);

    if (ch != EOF)
    {
        ungetc(ch, stdin);
        return 1;
    }

    return 0;
#elif defined(_WIN32) || defined(_WIN64)
    return _kbhit();
#endif
}

bool Hand::grasp()
{
    // This function checks the current state of the hand and then closes the fingers if they are open

    bool state = false; //true means hand is closed
    //change above init to true once get_state function is ready
    //state = get_hand_state();
    int f = Hand::group_move();
    cout << f;
    // if(!state)
    // {
    //     if(!close_all()){
    //         cout <<"ERROR: Could not close hand!" <<endl;
    //     }
    //     else{
    //         cout <<"SUCCESS: Hand closed!" <<endl;
    //     }
    // }
    // else{
    //     cout <<"ERROR: Hand is already closed!" <<endl;
    // }
    return true;
}
bool Hand::drop()
{
    // This function checks the current state of the hand and then opens the fingers if they are closed

    bool state = true; //true means hand is closed
    state = get_hand_state();

    if (state)
    {
        //move fingers
    }
    else
    {
        cout << "hand is already open" << endl;
    }
}
bool Hand::get_hand_state()
{
    //Create object of actuator class and read_pos of actuators to set hand state
}
bool Hand::finger_move()
{
    //This function will take a motor ID and move an individual finger
    return false;
}
int Hand::group_move()
{
    // This function makes use of the groupsync commands from the dynamixel sdk to move 3 motors all at once
    // If it is required to control other addresses in the control table for separate motors, either write them individually
    // or use the groupbulk commands from the dynamixel sdk to do so

    // Initialize PortHandler instance
    // Set the port path
    // Get methods and members of PortHandlerLinux or PortHandlerWindows
    dynamixel::PortHandler *portHandler = dynamixel::PortHandler::getPortHandler(DEVICENAME);

    // Initialize PacketHandler instance
    // Set the protocol version
    // Get methods and members of Protocol1PacketHandler or Protocol2PacketHandler
    dynamixel::PacketHandler *packetHandler = dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION);

    // Initialize GroupSyncWrite instance
    dynamixel::GroupSyncWrite groupSyncWrite(portHandler, packetHandler, ADDR_PRO_GOAL_POSITION, LEN_PRO_GOAL_POSITION);

    // Initialize Groupsyncread instance for Present Position
    dynamixel::GroupSyncRead groupSyncRead(portHandler, packetHandler, ADDR_PRO_PRESENT_POSITION, LEN_PRO_PRESENT_POSITION);

    int index = 0;
    int dxl_comm_result = COMM_TX_FAIL;                                                  // Communication result
    bool dxl_addparam_result = false;                                                    // addParam result
    bool dxl_getdata_result = false;                                                     // GetParam result
    int dxl_goal_position[2] = {DXL_MINIMUM_POSITION_VALUE, DXL_MAXIMUM_POSITION_VALUE}; // Goal position

    uint8_t dxl_error = 0; // Dynamixel error
    uint8_t param_goal_position[4];
    int32_t dxl1_present_position = 0, dxl2_present_position = 0, dxl3_present_position = 0; // Present position

    // Open port
    if (portHandler->openPort())
    {
        printf("Succeeded to open the port!\n");
    }
    else
    {
        printf("Failed to open the port!\n");
        printf("Press any key to terminate...\n");
        getch();
        return 0;
    }

    // Set port baudrate
    if (portHandler->setBaudRate(BAUDRATE))
    {
        printf("Succeeded to change the baudrate!\n");
    }
    else
    {
        printf("Failed to change the baudrate!\n");
        printf("Press any key to terminate...\n");
        getch();
        return 0;
    }

    // Enable Dynamixel 2 Torque
    dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, DXL2_ID, ADDR_PRO_TORQUE_ENABLE, TORQUE_ENABLE, &dxl_error);
    if (dxl_comm_result != COMM_SUCCESS)
    {
        cout << "[ID:00" << DXL2_ID << "] Torque Enable failed" << endl;
        //packetHandler->printTxRxResult(dxl_comm_result);
    }
    else if (dxl_error != 0)
    {
        //packetHandler->printRxPacketError(dxl_error);
    }
    else
    {
        printf("Dynamixel 2 has been successfully connected \n");
    }

    // Enable Dynamixel 3 Torque
    dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, DXL3_ID, ADDR_PRO_TORQUE_ENABLE, TORQUE_ENABLE, &dxl_error);
    if (dxl_comm_result != COMM_SUCCESS)
    {
        cout << "[ID:00" << DXL3_ID << "] Torque Enable failed" << endl;
        //packetHandler->printTxRxResult(dxl_comm_result);
    }
    else if (dxl_error != 0)
    {
        //packetHandler->printRxPacketError(dxl_error);
    }
    else
    {
        printf("Dynamixel 3 has been successfully connected \n");
    }

    // Enable Dynamixel 4 Torque
    dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, DXL4_ID, ADDR_PRO_TORQUE_ENABLE, TORQUE_ENABLE, &dxl_error);
    if (dxl_comm_result != COMM_SUCCESS)
    {
        cout << "[ID:00" << DXL4_ID << "] Torque Enable failed" << endl;
        //packetHandler->printTxRxResult(dxl_comm_result);
    }
    else if (dxl_error != 0)
    {
        //packetHandler->printRxPacketError(dxl_error);
    }
    else
    {
        printf("Dynamixel 4 has been successfully connected \n");
    }

    // Add parameter storage for Dynamixel 2 present position value
    dxl_addparam_result = groupSyncRead.addParam(DXL2_ID);
    if (dxl_addparam_result != true)
    {
        fprintf(stderr, "[ID:%03d] groupSyncRead addparam failed", DXL2_ID);
        return 0;
    }

    // Add parameter storage for Dynamixel 3 present position value
    dxl_addparam_result = groupSyncRead.addParam(DXL3_ID);
    if (dxl_addparam_result != true)
    {
        fprintf(stderr, "[ID:%03d] groupSyncRead addparam failed", DXL3_ID);
        return 0;
    }

    // Add parameter storage for Dynamixel 4 present position value
    dxl_addparam_result = groupSyncRead.addParam(DXL4_ID);
    if (dxl_addparam_result != true)
    {
        fprintf(stderr, "[ID:%03d] groupSyncRead addparam failed", DXL4_ID);
        return 0;
    }

    while (1)
    {
        printf("Press any key to continue! (or press ESC to quit!)\n");
        if (getch() == ESC_ASCII_VALUE)
            break;
        // Allocate goal position value into byte array
        param_goal_position[0] = DXL_LOBYTE(DXL_LOWORD(dxl_goal_position[index]));
        param_goal_position[1] = DXL_HIBYTE(DXL_LOWORD(dxl_goal_position[index]));
        param_goal_position[2] = DXL_LOBYTE(DXL_HIWORD(dxl_goal_position[index]));
        param_goal_position[3] = DXL_HIBYTE(DXL_HIWORD(dxl_goal_position[index]));

        // Add Dynamixel 2 goal position value to the Syncwrite storage
        dxl_addparam_result = groupSyncWrite.addParam(DXL2_ID, param_goal_position);
        if (dxl_addparam_result != true)
        {
            fprintf(stderr, "[ID:%03d] groupSyncWrite addparam failed", DXL2_ID);
            return 0;
        }

        // Add Dynamixel 3 goal position value to the Syncwrite parameter storage
        dxl_addparam_result = groupSyncWrite.addParam(DXL3_ID, param_goal_position);
        if (dxl_addparam_result != true)
        {
            fprintf(stderr, "[ID:%03d] groupSyncWrite addparam failed", DXL3_ID);
            return 0;
        }

        // Add Dynamixel 4 goal position value to the Syncwrite parameter storage
        dxl_addparam_result = groupSyncWrite.addParam(DXL4_ID, param_goal_position);
        if (dxl_addparam_result != true)
        {
            fprintf(stderr, "[ID:%03d] groupSyncWrite addparam failed", DXL4_ID);
            return 0;
        }

        // Syncwrite goal position
        dxl_comm_result = groupSyncWrite.txPacket();
        if (dxl_comm_result != COMM_SUCCESS)
        {
            printf("ERROR: Unable to write Goal Position!");
            //packetHandler->printTxRxResult(dxl_comm_result);
        }
        
        // Clear syncwrite parameter storage
        groupSyncWrite.clearParam();

        do
        {
            // Syncread present position
            dxl_comm_result = groupSyncRead.txRxPacket();
            if (dxl_comm_result != COMM_SUCCESS)
            {
                printf("ERROR: Unable to read Current Position!");
                //packetHandler->printTxRxResult(dxl_comm_result);
            }

            // Check if groupsyncread data of Dynamixel 2 is available
            dxl_getdata_result = groupSyncRead.isAvailable(DXL2_ID, ADDR_PRO_PRESENT_POSITION, LEN_PRO_PRESENT_POSITION);
            if (dxl_getdata_result != true)
            {
                fprintf(stderr, "[ID:%03d] groupSyncRead getdata failed", DXL2_ID);
                return 0;
            }

            // Check if groupsyncread data of Dynamixel 3 is available
            dxl_getdata_result = groupSyncRead.isAvailable(DXL3_ID, ADDR_PRO_PRESENT_POSITION, LEN_PRO_PRESENT_POSITION);
            if (dxl_getdata_result != true)
            {
                fprintf(stderr, "[ID:%03d] groupSyncRead getdata failed", DXL3_ID);
                return 0;
            }

            // Check if groupsyncread data of Dynamixel 4 is available
            dxl_getdata_result = groupSyncRead.isAvailable(DXL4_ID, ADDR_PRO_PRESENT_POSITION, LEN_PRO_PRESENT_POSITION);
            if (dxl_getdata_result != true)
            {
                fprintf(stderr, "[ID:%03d] groupSyncRead getdata failed", DXL4_ID);
                return 0;
            }
            // Get Dynamixel 2 present position value
            dxl1_present_position = groupSyncRead.getData(DXL2_ID, ADDR_PRO_PRESENT_POSITION, LEN_PRO_PRESENT_POSITION);

            // Get Dynamixel 3 present position value
            dxl2_present_position = groupSyncRead.getData(DXL3_ID, ADDR_PRO_PRESENT_POSITION, LEN_PRO_PRESENT_POSITION);

            // Get Dynamixel 3 present position value
            dxl3_present_position = groupSyncRead.getData(DXL4_ID, ADDR_PRO_PRESENT_POSITION, LEN_PRO_PRESENT_POSITION);

            printf("[ID:%03d] GoalPos:%03d  PresPos:%03d\tID:%03d] GoalPos:%03d  PresPos:%03d\t[ID:%03d] GoalPos:%03d  PresPos:%03d\n", DXL2_ID, dxl_goal_position[index], dxl1_present_position, DXL3_ID, dxl_goal_position[index], dxl2_present_position, DXL4_ID, dxl_goal_position[index], dxl3_present_position);

        } while ((abs(dxl_goal_position[index] - dxl1_present_position) > DXL_MOVING_STATUS_THRESHOLD) || (abs(dxl_goal_position[index] - dxl2_present_position) > DXL_MOVING_STATUS_THRESHOLD)|| (abs(dxl_goal_position[index] - dxl3_present_position) > DXL_MOVING_STATUS_THRESHOLD));

        // Change goal position
        if (index == 0)
        {
            index = 1;
        }
        else
        {
            index = 0;
        }
    }
    // Disable Dynamixel 2 Torque
    dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, DXL2_ID, ADDR_PRO_TORQUE_ENABLE, TORQUE_DISABLE, &dxl_error);
    if (dxl_comm_result != COMM_SUCCESS)
    {
        cout <<"ERROR:" <<DXL2_ID <<"Unable to disconnect motor" <<endl;
        //packetHandler->printTxRxResult(dxl_comm_result);
    }
    else if (dxl_error != 0)
    {
        //packetHandler->printRxPacketError(dxl_error);
    }

    // Disable Dynamixel 3 Torque
    dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, DXL3_ID, ADDR_PRO_TORQUE_ENABLE, TORQUE_DISABLE, &dxl_error);
    if (dxl_comm_result != COMM_SUCCESS)
    {
        cout <<"ERROR:" <<DXL3_ID <<"Unable to disconnect motor" <<endl;
        //packetHandler->printTxRxResult(dxl_comm_result);
    }
    else if (dxl_error != 0)
    {
        //packetHandler->printRxPacketError(dxl_error);
    }

    // Disable Dynamixel 4 Torque
    dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, DXL4_ID, ADDR_PRO_TORQUE_ENABLE, TORQUE_DISABLE, &dxl_error);
    if (dxl_comm_result != COMM_SUCCESS)
    {
        cout <<"ERROR:" <<DXL4_ID <<"Unable to disconnect motor" <<endl;
        //packetHandler->printTxRxResult(dxl_comm_result);
    }
    else if (dxl_error != 0)
    {
        //packetHandler->printRxPacketError(dxl_error);
    }
    // Close port
    portHandler->closePort();

    return 0;
}
