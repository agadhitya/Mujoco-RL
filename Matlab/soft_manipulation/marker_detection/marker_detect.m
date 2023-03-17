clear cam;
cam = webcam(1)
r = rateControl(30);
while(true)
   time = r.TotalElapsedTime;
   img = snapshot(cam);
   hold on;
   drawnow;
   detect_green(img);
   detect_red(img);
   waitfor(r);
end