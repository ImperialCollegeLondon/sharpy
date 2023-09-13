function state_space_converted = adjust_state_space_system(state_space_system, input_settings)
%convert_state_space_for_LQR Adjust extracted state space system
%   The loaded state space system exported from SHARPy is adjusted here for
%   the closed-loop simulations. This includes:
%       - extracts the column from the B and D matrices that describe the
%         effect of the gust input to the states
%       - removing all unused inputs and rearranging for example the
%         deflection of a control surface and its rate as both depend on
%         each other
%       - removes not considered outputs
%       - assembles updated discrete state space system


%% Reduce model by deleting unused outupts (only tip deflection relevant here)
C_sensor = state_space_system.C(input_settings.index.tip_displacement,:);
D_sensor = state_space_system.D(input_settings.index.tip_displacement,:);

%% Reduce model with not used inputs
idx_input_end =input_settings.index.control_input_start + 2  * input_settings.num_control_surfaces - 1;

idx_inputs = [input_settings.index.control_input_start:idx_input_end];
B_cs = state_space_system.B(:,idx_inputs);
D_cs = D_sensor(:,idx_inputs);  


%% Extract delta to state

new_A_colum = B_cs(:,1 :input_settings.num_control_surfaces);
A = [state_space_system.A new_A_colum; ...
    zeros(input_settings.num_control_surfaces,...
         (size(state_space_system.A,2)+ input_settings.num_control_surfaces))];

for counter = 0:1:input_settings.num_control_surfaces-1
    A(size(A,1)-counter,size(A,2)-counter) = 1; % Explain one
end

%Delete delta input and add delta_dot influence on delta 
B_cs(:,1:input_settings.num_control_surfaces) = [];
B_cs = [B_cs; eye(input_settings.num_control_surfaces) * state_space_system.Ts]; 

% Adding feed through of the delta-input on the output on C and deleting 
% column added in C out of D; new column
C = [C_sensor D_cs(:,1:input_settings.num_control_surfaces)];
D_cs(:, 1:input_settings.num_control_surfaces) = []; 


%% Get Disturbance Matrices
G = [state_space_system.B(:,input_settings.index.gust_input_start); ...
    zeros(input_settings.num_control_surfaces, 1)];
H = D_sensor(:,input_settings.index.gust_input_start);
%% Assemble final state space system

state_space_converted = ss(A,[B_cs G],C,[D_cs H],state_space_system.Ts);

end