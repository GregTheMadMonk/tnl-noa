\page tutorial_Linear_solvers  Linear solvers tutorial

[TOC]


# Introduction

Solvers of linear systems are one of the most important algorithms in scientific computations. TNL offers the followiing iterative methods:

1. Stationary methods
   1. [Jacobi method](https://en.wikipedia.org/wiki/Jacobi_method) (\ref TNL::Solvers::Linear::Jacobi)
   2. [Successive-overrelaxation method, SOR]([https://en.wikipedia.org/wiki/Successive_over-relaxation]) (\ref TNL::Solvers::Linear::SOR) - CPU only currently
2. Krylov subspace methods
   1. [Conjugate gradient method, CG](https://en.wikipedia.org/wiki/Conjugate_gradient_method) (\ref TNL::Solvers::Linear::CG)
   2. [Biconjugate gradient stabilized method, BICGStab](https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method) (\ref TNL::Solvers::Linear::BICGStab)
   3. [Biconjugate gradient stabilized method, BICGStab(l)](https://dspace.library.uu.nl/bitstream/handle/1874/16827/sleijpen_93_bicgstab.pdf) (\ref TNL::Solvers::Linear::BICGStabL)
   4. [Transpose-free quasi-minimal residual method, TFQMR]([https://second.wiki/wiki/algoritmo_tfqmr]) (\ref TNL::Solvers::Linear::TFQMR)
   5. [Generalized minimal residual method, GMRES](https://en.wikipedia.org/wiki/Generalized_minimal_residual_method) (\ref TNL::Solvers::Linear::GMRES) with various methods of orthogonalization
      1. Classical Gramm-Schmidt, CGS
      2. Classical Gramm-Schmidt with reorthogonalization, CGSR
      3. Modified Gramm-Schmidt, MGS
      4. Modified Gramm-Schmidt with reorthogonalization, MGSR
      5. Compact WY form of the Householder reflections, CWY

The Krylov subspace methods can be combined with the following precoditioners:

1. Jacobi
2. ILU - CPU only currently

# Iterative solvers of linear systems

All iterative solvers for linear systems can be found in the namespace \ref TNL::Solvers::Linear. The following example shows the use the iterative solvers:

\includelineno Solvers/Linear/IterativeLinearSolverExample.cpp

The result looks as follows:

\include IterativeLinearSolverExample.out