#!/bin/bash

rostopic echo -b freq_test.bag -p /base_pos >~/owi_data/raw_data/base_px.csv
rostopic echo -b freq_test.bag -p /pixel3 >~/owi_data/raw_data/px3.csv
rostopic echo -b freq_test.bag -p /newFrame >~/owi_data/raw_data/flag.csv
rostopic echo -b freq_test.bag -p /pwm >~/owi_data/raw_data/input_vel.csv

mv freq_test.bag ~/owi_data/raw_data/
