function gust_time_series = get_1minuscosine_gust_input(gust_length, gust_intensity, dt, u_inf, simulation_time)
%get_1minuscosine_gust_input Gust input is generated and stored.

gust_time = [0:dt:simulation_time];
offset_gust = 0;
gust_intensity = gust_intensity*u_inf;

end_gust = gust_length/u_inf;
x = [0:dt:end_gust]*u_inf; % spatial coordinate

gust_cos = (1.0 - cos(2*pi*x/ gust_length)) * gust_intensity / 2;
gust_vel_z = [zeros(1,offset_gust) gust_cos];

%% Adjust vector length 
delta_vector_length = size(gust_time,2) - size(gust_vel_z,2);
gust_vel_z = [gust_vel_z zeros(1, delta_vector_length)]; 

%% Store gust velocities in time history
gust_time_series = [gust_time; gust_vel_z]';

end