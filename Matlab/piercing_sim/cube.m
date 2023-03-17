classdef cube
    
    properties
        size; u_s; u_r; mass; F; mg; piercing_angles;
    end  % properties
    
    methods
%% Constructor
        function obj = cube()
            obj.size = 2;
            obj.u_s = 0.25;
            obj.u_r = 0.1;
            obj.mass = 0.05;
            obj.F = 1;
            obj.mg = obj.mass*10;
            obj.piercing_angles = [15,21,30,45,60,90];
        end  % function
%% Generate Force Function
        function vectors = generate_force(obj)
         for i = 1:length(obj.piercing_angles)
             vec = [sind(obj.piercing_angles(i)), 0 , -cosd(obj.piercing_angles(i))];
             vectors(i,:) = vec;
         end
        end  % function
%% Food Plate Interaction Simulation Function        
        function food_plate_simulator(obj,vectors)
            force_vectors = obj.F*vectors; % Decompose force vectors along co-ordinate axes
            %% Compute forces
            
            % Recording Video
            v = VideoWriter('trial.avi')
            v.FrameRate = 2
            open(v)
            
            for i = 1:length(force_vectors)
                static_friction = [-obj.u_s*obj.mg+obj.u_s*force_vectors(i,3),0,0]; %Assuming friction is always in -x since block is always pushed in +x
            if(force_vectors(i,1)>-static_friction(1))
                % Cube Slides
                sliding_friction = [-obj.u_r*obj.mg+obj.u_r*force_vectors(i,3),0,0]; % now under effect of sliding friction
                resultant_force = force_vectors(i) + sliding_friction;
                acceleration = resultant_force/obj.mass;
            else
                % Cube does not slide
                acceleration = [0,0,0];
            end
            %% Simulate motion of cube for t = 0 to t = 1
            initial_vel = 0; acceleration_x = acceleration(1); end_time = 1;
            for time = 0:0.2:end_time
                clf('reset');
                figure(1)%'visible','off')
                %set(gcf,'Position',[0, 0, 1000,1000])
                title("Trial:"+i)
                axis([0 40 -15 15 0 20])
                grid on
                hold on

                s_x = initial_vel*time + 0.5*acceleration_x*(time^2);
                initial_vel = initial_vel + acceleration_x*time;

                center_cube = [s_x, 1, 2];
                q = quiver3(center_cube(1),center_cube(2),center_cube(3),vectors(i,1),vectors(i,2),vectors(i,3));
                q.LineWidth = 5;
                plotcube((obj.size*[1,1,1]),[s_x,0,0],0.8,[1,0,0]);
                F = getframe(gcf);
                writeVideo(v,F)
                %pause(.5)
            end
            end
            close(v)
            %figure()
            %movie(F,1,2)
        end  % function
%% Max Piercing Angle Function
        function max = max_piercing_angle(obj,u)
            slide = false; u_s = u;
            piercing_angle = 0; max = 0;
            % Need Cube and force properties

            while(~slide)
                % Decompose Forces ie Calculate Force Vectors
                vec = [sind(piercing_angle), 0 , -cosd(piercing_angle)];
                force_vec = obj.F*vec;
                % Calculate Static Friction
                static_friction = [-u_s*obj.mg+u_s*force_vec(3),0,0]; %Assuming friction is always in -x since block is always pushed in +x
                if(force_vec(1)>-static_friction(1))
                    slide = true;
                else
                    max = piercing_angle;
                    piercing_angle = piercing_angle + 0.25;
                end
            end
        
        end  % function
%% Max force before Sliding
           function max_f = max_force(obj,p)
           max_f = 0; slide = false; piercing_angle = p;
           while(~slide)
                   max_f = max_f + 0.01;
               % Decompose Forces ie Calculate Force Vectors
               vec = [sind(piercing_angle), 0 , -cosd(piercing_angle)];
               force_vec = max_f*vec;
               % Calculate Static Friction
               static_friction = [-obj.u_s*obj.mg+obj.u_s*force_vec(3),0,0]; %Assuming friction is always in -x since block is always pushed in +x
               if(force_vec(1)<-static_friction(1))
                   continue;
               else
                   slide = true;
               end  
           end  % while
           end  % function
%% Generate force vectors for Food Fork Interaction
function vectors = food_fork_forces(obj)
    for i = 1:length(obj.piercing_angles)
        vec_obj_frame = [cosd(piercing_angles(i)),0,-sind(obj.piercing_angles(i))]
        vectors(i,:) = vec_obj_frame;
    end  % for loop
end  % function
%% Food Fork Interaction function
function fall = fall_checker(obj,vector,u)
    fall = true; u_s = u;
    force_vector = obj.mg*vector;
        static_friction = [u_s*force_vector(3),0,0];
        if(force_vector(1)>-static_friction(1))
            % cube falls
            fall = true;
        else
            % cube does not fall
            fall = false;
        end  % if-else
end  % function
    end  % methods
end  % classdef