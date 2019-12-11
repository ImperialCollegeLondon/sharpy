---
layout: post
title: 'Mass and stiffness effect on damping'
date: 2019-12-10
author: Norberto Goizueta
---

We investigate the effect of changing the length of a clamped wing (keeping all structural properties constant) on
the damping provided by aerodynamic forces.

By maintaining the same structural properties (stiffness and inertia), what we are changing essentially by increasing
the length of the clamped with is moving from a stiffness dominated problem (short and stiff wing) to a much more
inertia dominated system (long and slender).

The wing to be tested is a modified Goland wing with varying length, from 1m to 45m span, submerged in a 25m/s 
uniform, steady flow and with gravitational forces included.

The nonlinear equilibrium condition is computed for each test and the problem is linearised about it. Thence, the
aerodynamic system is reduced using a Krylov-subspace method (to analyse the stability with greater ease). The
reduced aerodynamic system is coupled with a linearised structural system of the beam projected onto a few mode shapes.

A discrete-time, linear time invariant system is the result of this process. Therefore, by plotting the eigenvalues of
the plant matrix (transformed into continuous time) we are able to analyse the stability of said system. The root
locus diagram and damping ratio evolution are shown below with the wing increasing in span (light to dark).

![Root Locus of Coupled Aeroelastic System](/sharpy/assets/img/2019_12_11/root_locus_n4_length.png)
_Light to Dark with increasing wing span_

![Damping Ratio](/sharpy/assets/img/2019_12_11/damping_ratio_length.png)


