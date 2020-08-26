# Frequently Asked Questions [FAQs]

Over the years, we have gathered a valuable experience running SHARPy so we would like to collect here a few of the
most frequently asked questions we get here to hopefully help other users.

In addition to the questions listed below, do check the [issues](https://github.com/ImperialCollegeLondon/sharpy/issues?q=is%3Aissue) page in our GitHub repo as you may find useful 
information there. Do also check our [Short Debugging Guide](./debug.html).

* __[Q] I get a `ModuleNotFound Error` when trying to run SHARPy.__

    _[A]_ Make sure you have loaded the SHARPy variables using the command `source <path_to_sharpy>/bin_sharpy_var.sh`.
    
* __[Q] When plotting the aerodynamic forces in Paraview from the UVLM, the forces at the boundary between two 
surfaces (for instance at the wing root) appears halved. Is my simulation incorrect?__

    _[A]_ This is most likely not an issue with your simulation. We have observed over time that Paraview actually only
    plots the result from one of the surfaces, hence why it appears half of what it should be. If you extract the forces
    without using Paraview using the 
    [WriteVariablesTime](https://ic-sharpy.readthedocs.io/en/master/includes/postprocs/WriteVariablesTime.html) postprocessor you will get the correct result.
     
* __[Q] My time-domain simulation does not converge. I get a `SolverNotConverged` error. What can I do?__

    _[A]_ This is quite an open question and it could be for a wide variety of reasons. Things that should be in your 
    First Aid kit for these situations:
    
    - Is your tolerance appropriate? If you raise the tolerance to something (maybe unreasonably) high does it 
    converge?
    
    - Is your number of iterations sufficient? Increase the number of maximum allowed iterations.
    
    - Is there anything happening at all in your simulation? We have all fallen into the trap of trying to run 
    a time domain simulation of something that is already in steady state. I.e. you calculate its static
    equilibrium and then try to advance in time. Since nothing is happening the convergence criteria in the solvers
    may not be triggered and reach the maximum number of iterations. Solution: make sure something happens in your time
    domain simulation: gusts, external forces, control surface deflections...
    
    - Are you giving your simulation too much of a "kick"? Sometimes we simulate things that dramatically change the
    state of the problem from one time step to another (like adding very large external forces at once) which may lead
    to trouble. You can choose to load the forces progressively by increasing the `num_load_steps` setting in our
    structural solvers.
    
Hopefully this list will grow over time with some of the common questions previous users encounter. If you cannot solve
your problem please open an [issue](https://github.com/ImperialCollegeLondon/sharpy/issues) on Github and assign it the
label `label:question` so we can keep track of it and others can benefit of the discussion.