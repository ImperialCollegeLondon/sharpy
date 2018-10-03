The ``Algebra`` package
-----------------------
This package includes routines used often during geometry generation.

Tangent vector calculation
++++++++++++++++++++++++++
Method:
    1) A n_nodes-1 polynomial is fitted through the nodes per dimension.
    2) Those polynomials are analytically differentiated with respect to the node index
    3) The tangent vector is given by:

.. math:: \vec{t} = \frac{s_x'\vec{i} + s_y'\vec{j} + s_z'\vec{k}}{\left| s_x'\vec{i} + s_y'\vec{j} + s_z'\vec{k}\right|}


where :math:`'` notes the differentiation with respect to the index number

.. automodule:: sharpy.utils.algebra
    :members: tangent_vector
