# Package for [OWI 535 Robotic Arm Edge](https://www.amazon.com/OWI-Robotic-Soldering-Required-Extensive/dp/B0017OFRCY/)

## About the Package
This package interfaces with [OWI USB Interface](https://www.amazon.com/OWI-USB-Interface-Robotic-Arm/dp/B0028MBWS2) to command the robot. The following is a list of available services and their short descriptions. For a full explanation of how to use the service check the section below.

* writecmd.srv - this service writes commands to the robot
* generatecmd.srv - this service generates 3 byte commands for the robot

## Service Documentation

### writecmd.srv

This service sends a 3 byte command to the OWI USB Interface. To try this out, connect the USB to your computer, turn on the power switch for the USB kit and run the *cmd_robot_node*. To do this run roscore in a terminal.

```
roscore
```

In a new terminal rosrun the node. **Dont forget to source your ws if you don't have it setup in your bashrc!**

```
rosrun cmd_robot_node
```

In a new terminal call the service. **Dont forget to source your ws if you don't have it setup in your bashrc!**

```
rosservice call cmdOwiServer "Str1:'00' Str2:'00' Str3:'01'"
```

This should turn on the LED on the robot. You can set Str3 to '00' to turn off the LED.

#### Permission Denied Error
```
libusb: error [_get_usbfs_fd] libusb couldn't open USB device /dev/bus/usb/001/025: Permission denied
libusb: error [_get_usbfs_fd] libusb requires write access to USB device nodes.
Error opening device
```

If you receive an error with write permissions to the USB port use the [chmod](https://wiki.linuxquestions.org/wiki/Chmod) command as shown below. This will elevate permissions for all users and the group on the computer to write to the USB device you receive an error for as in the example below (I received an error for USB001/025). *Make sure to change the USB device ID in the command below to reflect the error!*

```
sudo chmod 666 /dev/bus/usb/001/025
```
### generatecmd.srv

This service generates the 3 byte command for writing to the OWI USB Interface. It requires the inidividual motors and their directions of rotations as inputs. It will then generate a 3 byte cmd to run those motors. This command can be sent to the writecmd.srv to run the robot. An additional PWM service can be added in between to control the velocity of the joints. Here, a command table for each motor can be defined using bitset, the service will convert each 8 bit command into byte commands for the arm. This is done so we can run combinations of multiple motors by "adding"(|) their bit commands and then converting them into byte commands to write to the robot.  

### pwm.srv

This service can drive a single software pwm when called. It is a blocking service, which is why we are moving away from the software pwm architecture and using a dedicated hardware pwm controller instead.

## Publishers

This package also exposes publishers to a topic which can be used to send messages to external nodes that run have hardware PWM capability to control the robot's individual joints.