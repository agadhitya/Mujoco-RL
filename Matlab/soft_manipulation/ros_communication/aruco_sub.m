% Connect to ROSMASTER
rosinit('http://localhost:11311/');

% Subscriber object
sub = rossubscriber('/aruco_simple/pose');
% Wait for msg
msg = receive(sub,10)


% Check for a ros spin()

% Shutdown
rosshutdown