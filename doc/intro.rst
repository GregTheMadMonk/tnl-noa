============
Introduction
============

TNL means *Template Numerical Library*. Aim of this project is to develop *efficient, flexible and easy to use* numerical library.

**Efficiency**
   Complex numerical simulations may take hundreds of hours. Fast and efficient solvers are therefore very important. TNL is designed to profit from abilities of new accelerators like GPUs (NVidia GeForce, Tesla) and MICs (Xeon Phi). To generate efficient executables, we avoid use of virtual methods on low levels of the code. Instead, C++ templates are used. 

**Flexibility**
   Development of new numerical schemes and solvers often requires to test many different approaches. Thanks to C++ templates and the design of TNL, it should be quite easy to switch between different schemes, solvers, meshes, precision of the floating point arithmetics or parallel architectures.

**Easy to use**
   Thanks to C++ templates, TNL offers automatic set-up of underlying structures (numerical meshes, sparse matrices, etc.), solvers (linear solvers, Runge-Kutta solvers, PDE solvers) and parallel architectures (GPU, MIC or MPI (not implemented yet)). TNL can also manage configuration parameters passed from the command line. The user may then concentrate only on the numerical model. 

:Authors:
   **Tomáš Oberhuber** - TNL design

   **Vítězslav Žabka** - unstructured numerical mesh

   **Vladimír Klement** - multigrid methods

   **Tomáš Sobotík** - numerical methods for signed distance function

   **Ondřej Székely** - FDM solvers for non-linear diffusion problems

   **Libor Bakajsa** - sparse matrix formats for GPUs

   **Jan Vacata** - sparse matrix formats for GPUs

   **Martin Heller** - sparse matrix formats for GPUs

   **Matěj Novotný** - quad double arithmetics