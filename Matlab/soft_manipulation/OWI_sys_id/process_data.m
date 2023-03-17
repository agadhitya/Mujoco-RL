function[base_marker, ee_marker, shoulder_marker, flag, input_vel, elbow_marker] = process_data(base_marker, ee_marker, shoulder_marker, flag, input_vel, elbow_marker, posn)

    base_marker = base_marker(posn:length(base_marker),:);
    ee_marker = ee_marker(posn:length(ee_marker),:);
    shoulder_marker = shoulder_marker(posn:length(shoulder_marker),:);
    flag = flag(posn:length(flag),:);
    input_vel = input_vel(posn:length(input_vel),:);
    elbow_marker = elbow_marker(posn:length(elbow_marker),:);
end