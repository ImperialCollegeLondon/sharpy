# The SHARPy Case files

##  The FEM file

The case.fem.h5 file has several components. We go one by one:

   - `num_node_elem [int]`: number of nodes per element.
   
        Always 3 in our case (3 nodes per structural elements - quadratic beam elements).
    
   - `num_elem [int]`: number of structural elements.
    
   - `num_node [int]`: number of nodes. 
   
       For simple structures, it is `num_elem*(num_node_elem - 1) - 1`. 
       For more complicated ones, you need to calculate it properly.

   - `coordinates [num_node, 3]` coordinates of the nodes in body-attached FoR.

   - `connectivites [num_elem, num_node_elem]`: Beam element's connectivities. 
   
       Every row refers to an element, and the three integers in that row are the indices of the three nodes 
       belonging to that elem. Now, the catch: the ordering is not as you'd think. Order them as `[0, 2, 1]`. 
       That means, first one, last one, central one. The following image shows the node indices inside the 
       circles representing the nodes, the element indices in blue and the resulting connectivities matrix next to it. 
       Connectivities are tricky when considering complex configurations. Pay attention at the beginning and you'll 
       save yourself a lot of trouble.


```
stiffness_db [:, 6, 6] database of stiffness matrices. The first dimension has as many elements as different stiffness matrices are in the model.

elem_stiffness [num_elem] array of indices (starting at 0). Basically, it links every element (index) to the stiffness matrix index in stiffness_db. For example elem_stiffness[0] = 0; elem_stiffness[2] = 1 means that the element 0 has a stiffness matrix equal to stiffness_db[0, :, :], and the second element has a stiffness matrix equal to stiffness_db[1, :, :].

The shape of a stiffness matrix, $\mathrm{S}$ is:
S=⎡⎣⎢⎢⎢⎢⎢⎢⎢⎢⎢EAGAyGAzGJEIyEIz⎤⎦⎥⎥⎥⎥⎥⎥⎥⎥⎥
 
with the cross terms added if needed.

mass_db and elem_mass follow the same scheme than the stiffness, but the mass matrix is given by:
M=[mIξ~cgm−ξ~cgmJ]
 
where $m$ is the distributed mass per unit length [kg/m], $\tilde{\bullet}$ is the skew-symmetric matrix of a vector and $\xi_{cg}$ is the location of the centre of gravity with respect to the elastic axis in MATERIAL (local) FoR.
And what is the Material FoR? This is an important point, because all the inputs that move WITH the beam are in material FoR. For example: follower forces, stiffness, mass, lumped masses...



The material frame of reference is noted as $B$. Essentially, the $x$ component is tangent to the beam in the increasing node ordering, $z$ looks up generally and $y$ is oriented such that the FoR is right handed.

In the practice (vertical surfaces, structural twist effects...) it is more complicated than this. The only sure thing about $B$ is that its $x$ direction is tangent to the beam in the increasing node number direction. However, with just this, we have an infinite number of potential reference frames, with $y$ and $z$ being normal to $x$ but rotating around it. The solution is to indicate a for_delta, or frame of reference delta vector ($\Delta$).



Now we can define unequivocally the material frame of reference. With $x_B$ and $\Delta$ defining a plane, $y_b$ is chosen such that the $z$ component is oriented upwards with respect to the lifting surface.

From this definition comes the only constraint to $\Delta$: it cannot be parallel to $x_B$.

frame_fo_reference_delta [num_elem, num_node_elem, 3] contains the $\Delta$ vector in body-attached ($A$) frame of reference. As a rule of thumb:
Δ=⎧⎩⎨⎪⎪⎪⎪⎪⎪⎪⎪[−1,0,0],if right wing[1,0,0],if left wing[0,1,0],if fuselage[−1,0,0],if vertical fin
 
These rules of thumb only work if the nodes increase towards the tip of the surfaces (and the tail in the case of the fuselage).

structural_twist [num_elem, num_node_elem] is technically not necessary, as the same effect can be achieved with FoR_delta. CAUTION previous versions of SHARPy had structural twist defined differently:
structural_twist = np.zeros((num_node, num_node_elem)) # this is wrong now, and will trigger and error in SHARPy, change it!
structural_twist = np.zeros((num_elem, num_node_elem)) # this is right.
boundary_conditions [num_node] is an array of integers (np.zeros((num_node, ), dtype=int)) and contains all 0 EXCEPT FOR:

One node NEEDS to have a 1, this is the reference node. Usually, the first node has 1 and is located in [0, 0, 0]. This makes things much easier.
If the node is a tip of a beam (is not attached to 2 elements, but just 1), it needs to have a -1.
beam_number [num_elem] is another array of integers. Usually you don't need to modify its value. Leave it at 0.

app_forces [num_elem, 6] contains the applied forces app_forces[:, 0:3] and moments app_forces[:, 3:6] in a given node. Important points: the forces are given in Material FoR (check above). That means that in a symmetrical model, a thrust force oriented upstream would have the shape $[0, T, 0, 0, 0, 0]$ in the right wing, while the left would be $[0, -T, 0, 0, 0, 0]$. Likewise, a torsional moment for twisting the wing leading edge up would be $[0, 0, 0, M, 0, 0]$ for the right, and $[0, 0, 0, -M, 0, 0]$ for the left. But careful, because an out-of-plane bending moment (wing tip up) has the same sign (think about it).

lumped_mass [:] is an array with as many masses as needed (in kg this time). Their order is important, as more information is required to implement them in a model.

lumped_mass_nodes [:] is an array of integers. It contains the index of the nodes related to the masses given in lumped_mass in order.

lumped_mass_inertia [:, 3, 3] is an array of 3x3 inertial tensors. The relationship is set by the ordering as well.

lumped_mass_position [:, 3] is the relative position of the lumped mass wrt the node (given in lumped_masss_nodes) coordinates. ATTENTION: the lumped mass is solidly attached to the node, and thus, its position is given in Material FoR.
```