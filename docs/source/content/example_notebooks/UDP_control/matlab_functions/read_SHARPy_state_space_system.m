function [state_space_system, eta_ref] = read_SHARPy_state_space_system(file_name_sharpy, folder, complex)
%read_SHARPy_state_space_system Gets sate space model from SHARPy output

absolute_file_path = strcat(folder, file_name_sharpy);
% Reference Point and Timestep
eta_ref = h5read(absolute_file_path, '/linearisation_vectors/eta');
dt = h5read(absolute_file_path, '/ss/dt');
% Read state space matrices 
if complex
    A = h5read(absolute_file_path, '/ss/A').r; 
    B = h5read(absolute_file_path, '/ss/B').r; 
    C = h5read(absolute_file_path, '/ss/C').r;
    D = h5read(absolute_file_path, '/ss/D');
else
    A = h5read(absolute_file_path, '/ss/A'); 
    B = h5read(absolute_file_path, '/ss/B'); 
    C = h5read(absolute_file_path, '/ss/C');
    D = h5read(absolute_file_path, '/ss/D');
end

% Sometimes matrices are exported transposed
if size(A, 1) ~= size(B, 1)
   A = transpose(A); 
   B = transpose(B); 
end
if size(C, 1) ~= size(D, 1)
   C = transpose(C); 
   D = transpose(D); 
end

% Assemble final discrete state space system
state_space_system = ss(A, B, C, D, dt);

end 
