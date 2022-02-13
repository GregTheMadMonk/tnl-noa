# ODE solvers tutorial

[TOC]

## Introduction

In this part, we describes solvers [ordinary differntial equations](https://en.wikipedia.org/wiki/Ordinary_differential_equation). TNL offers the following ODE solvers:

1. \ref TNL::Solvers::ODE::Euler - the Euler method with the 1-st order of accuracy.
2. \ref TNL::Solvers::ODE::Merson - the Runge-Kutta-Merson solver with the 4-th order of accuracy and adaptive choice of the time step.

Each solver has its static counterpart which can be run even in the GPU kernels which means that it can be combined with \ref TNL::Algorithms::ParallelFor for example. The static ODE solvers are the following:

1. \ref TNL::Solvers::ODE::StaticEuler - the Euler method with the 1-st order of accuracy.
2. \ref TNL::Solvers::ODE::StaticMerson - the Runge-Kutta-Merson solver with the 4-th order of accuracy and adaptive choice of the time step.

## Static ODE solvers



## Non-static ODE Solvers

### Basic setup


### Setup with a solver monitor

Solution of large linear systems may take a lot of time. In such situations, it is useful to be able to monitor the convergence of the solver of the solver status in general. For this purpose, TNL offers solver monitors. The solver monitor prints (or somehow visualizes) the number of iterations, the residue of the current solution approximation or some other metrics. Sometimes such information is printed after each iteration or after every ten iterations. The problem of this approach is the fact that one iteration of the solver may take only few milliseconds but also several minutes. In the former case, the monitor creates overwhelming amount of output which may even slowdown the solver. In the later case, the user waits long time for update of the solver status. The monitor in TNL rather runs in separate thread and it refreshes the status of the solver in preset time periods. The use of the iterative solver monitor is demonstrated in the following example.

\includelineno Solvers/Linear/IterativeLinearSolverWithMonitorExample.cpp

On the lines 1-70, we setup the same linear system as in the previous example, we create an instance of the Jacobi solver and we pass the matrix of the linear system to the solver. On the line 71, we set the relaxation parameter \f$ \omega \f$ of the Jacobi solver to 0.0005 (\ref TNL::Solvers::Linear::Jacobi). The reason is to slowdown the convergence because we want to see some iterations in this example. Next we create an instance of the solver monitor (lines 76 and 77) and we create a special thread for the monitor (line 78, \ref TNL::Solvers::SolverMonitorThread ). We set the refresh rate of the monitor to 10 milliseconds (line 79, \ref TNL::Solvers::SolverMonitor::setRefreshRate). We set a verbosity of the monitor to 1 (line 80 \ref TNL::Solvers::IterativeSolverMonitor::setVerbosity ). Next we set a name of the solver stage (line 81, \ref TNL::Solvers::IterativeSolverMonitor::setStage). The monitor stages serve for distinguishing between different phases or stages of more complex solvers (for example when the linear system solver is embedded into a time dependent PDE solver). Next we connect the solver with the monitor (line 82, \ref TNL::Solvers::IterativeSolver::setSolverMonitor). Finally we start the solver (line 83, \ref TNL::Solvers::Linear::Jacobi::start) and when the solver finishes we have to stop the monitor (line 84, \ref TNL::Solvers::SolverMonitor::stopMainLoop).

The result looks as follows:

\include IterativeLinearSolverWithMonitorExample.out

The monitoring of the solver can be improved by time elapsed since the beginning of the computation as demonstrated in the following example:

\includelineno Solvers/Linear/IterativeLinearSolverWithTimerExample.cpp

The only changes happen on lines 83-85 where we create an instance of TNL timer (line 83, \ref TNL::Timer), connect it with the monitor (line 84, \ref TNL::Solvers::SolverMonitor::setTimer) and start the timer (line 85, \ref TNL::Timer::start).

The result looks as follows:

\include IterativeLinearSolverWithTimerExample.out

### Choosing the solver and preconditioner type at runtime

When developing a numerical solver, one often has to search for a combination of various methods and algorithms that fit given requirements the best. To make this easier, TNL offers choosing the type of both linear solver and preconditioner at runtime by means of functions \ref TNL::Solvers::getLinearSolver and \ref TNL::Solvers::getPreconditioner. The following example shows how to use these functions:

\includelineno Solvers/Linear/IterativeLinearSolverWithRuntimeTypesExample.cpp

We still stay with the same problem and the only changes can be seen on lines 66-70. We first create an instance of shared pointer holding the solver (line 66, \ref TNL::Solvers::getLinearSolver) and the same with the preconditioner (line 67, \ref TNL::Solvers::getPreconditioner). The rest of the code is the same as in the previous examples with the only difference that we work with the pointer `solver_ptr` instead of the direct instance `solver` of the solver type.

The result looks as follows:

\include IterativeLinearSolverWithRuntimeTypesExample.out
