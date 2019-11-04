Calculate derivatives of induced velocity.
++++++++++++++++++++++++++++++++++++++++++

Calculate derivatives of induced velocity.

Methods:

- eval_seg_exp and eval_seg_exp_loop: profide ders in format 
	[Q_{x,y,z},ZetaPoint_{x,y,z}]
  and use fully-expanded analytical formula.
- eval_panel_exp: iterates through whole panel

- eval_seg_comp and eval_seg_comp_loop: profide ders in format 
	[Q_{x,y,z},ZetaPoint_{x,y,z}]
  and use compact analytical formula.


.. toctree::
	:glob:

	./Dvcross_by_skew3d
	./eval_panel_comp
	./eval_panel_cpp
	./eval_panel_exp
	./eval_panel_fast
	./eval_panel_fast_coll
	./eval_seg_comp_loop
	./eval_seg_exp
	./eval_seg_exp_loop
