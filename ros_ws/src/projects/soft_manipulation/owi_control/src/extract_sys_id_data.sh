#!/bin/bash

rostopic echo -b freq_test.bag -p /base_pos >~/owi_data/sys_id_data/base_px.csv
rostopic echo -b freq_test.bag -p /pixel3 >~/owi_data/sys_id_data/shoulder.csv
rostopic echo -b freq_test.bag -p /end_effector >~/owi_data/sys_id_data/ee.csv
rostopic echo -b freq_test.bag -p /pixel4 >~/owi_data/sys_id_data/elbow.csv
rostopic echo -b freq_test.bag -p /newFrame >~/owi_data/sys_id_data/flag.csv
rostopic echo -b freq_test.bag -p /pwm >~/owi_data/sys_id_data/input_vel.csv

mv freq_test.bag ~/owi_data/sys_id_data/