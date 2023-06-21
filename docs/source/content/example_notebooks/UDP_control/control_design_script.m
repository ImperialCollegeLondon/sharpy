% Control design script
%
clear all
close all
addpath('./matlab_functions/');
%% Define parameters
case_name = 'pazy_ROM'
output_folder = strcat('./output/',case_name)
output_folder_linear = strcat(output_folder, '/linear_results/');
complex_matrices = true % explanation

%% Reade from SHARPy generated state space model
[state_space_system, eta_ref] = read_SHARPy_state_space_system(...
                                        strcat(case_name, '.linss.h5'), ...
                                        strcat(output_folder, '/savedata/'), ...
                                        complex_matrices...
                                        );

%% Load simulation settings and store in struct
load(strcat(output_folder_linear, 'simulation_parameters.mat'))
n_nodes = uint8(n_nodes);
rigid_body_motions = false;

input_settings =  set_input_parameters(u_inf, ...
                                      state_space_system.Ts, ...
                                      num_modes, ...
                                      num_aero_states, ...
                                      rigid_body_motions, ...
                                      simulation_time, ...
                                      num_control_surfaces, ...
                                      n_nodes, ...
                                      control_input_start, ...
                                      gust_input_start);

%% Remove unused input and outputs from the generated ROM in SHARPy and link deflection and its rate

state_space_system = adjust_state_space_system(state_space_system, input_settings);


%% Get gust input

gust = get_1minuscosine_gust_input(...
    gust_length, ...
    gust_intensity, ...
    state_space_system.Ts, ...
    u_inf, ...
    simulation_time);

%% Run Simulink with PID Controller
P_gain = 0;
I_gain = 0;
D_gain = 0;

% open loop
controller_on = 0;
out_open_loop = sim("PID_linear_model.slx");
% closed loop P = 5
controller_on = 1;
P_gain = 5;
out_PID_5 = sim("PID_linear_model.slx");

% closed loop P = 10
P_gain = 10;
out_PID_10 = sim("PID_linear_model.slx");

%% Save results
deflection = [out_PID_5.delta.Data(:,1)...
    out_PID_10.delta.Data(:,1)...
    zeros(size(out_PID_5.delta.Time, 1), 1)];
deflection_rate = [out_PID_5.delta_dot.Data(:,1)...
    out_PID_10.delta_dot.Data(:,1)...
    zeros(size(out_PID_5.delta_dot.Time, 1), 1)];
tip_deflection = [out_PID_5.actual_output.Data(:,1)...
    out_PID_10.actual_output.Data(:,1)...
    out_open_loop.actual_output.Data(:,1)];
tip_deflection=  tip_deflection + eta_ref(n_nodes/2*6-3);
time_array = out_open_loop.tout;

output_folder_linear = strcat(output_folder, '/linear_results/');
writematrix(deflection, strcat(output_folder_linear, 'deflection.txt'),'Delimiter',',');
writematrix(time_array, strcat(output_folder_linear, 'time_array_linear.txt'),'Delimiter',',');
writematrix(deflection_rate,strcat(output_folder_linear, './deflection_rate.txt'),'Delimiter',',');
writematrix(tip_deflection,strcat(output_folder_linear, './tip_deflection.txt'),'Delimiter',',');
