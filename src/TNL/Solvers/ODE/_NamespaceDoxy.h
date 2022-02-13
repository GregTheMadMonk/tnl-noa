// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

namespace TNL {
    namespace Solvers {

    /**
     * \brief Namespace for solvers of ordinary differential equations.
     * 
     * Solvers in this namespace represent numerical methods for the solution of the (ordinary differential equations)[https://en.wikipedia.org/wiki/Ordinary_differential_equation]:
     * 
     * \f$ \frac{d \vec x}{dt} = \vec f( t, \vec x) \text{ on } (0,T) \f$
     * 
     * \f$ \vec x( 0 )  = \vec x_{ini}.
     * 
     * The following solvers are available:
     * 
     * 1. First order of accuracy
     *      a. \ref StaticEuler, \ref Euler - the Euler method
     * 2. Fourth order of accuracy
     *      a. \ref StaticMerson, \ref Merson - the Runge-Kutta-Merson method with adaptive choice of the time step.
     * 
     * The static variants of the solvers are supposed to be used when the unknown \f$ x \in R^n \f$ is expressed by a \ref Containers::StaticVector or it is a scalar, i.e.
     * \f$ x \in R \f$ expressed by a numeric type like `double` or `float`.
     */
    namespace ODE {} // namespace ODE
    }  // namespace Solvers
}  // namespace TNL
