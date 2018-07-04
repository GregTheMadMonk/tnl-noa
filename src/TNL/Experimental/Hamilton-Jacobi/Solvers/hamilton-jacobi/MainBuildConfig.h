/***************************************************************************
                          HamiltonJacobiBuildConfigTag.h  -  description
                             -------------------
    begin                : Jul 7, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Solvers/BuildConfigTags.h>

//namespace TNL {
//namespace Solvers {   

using namespace TNL::Solvers;

class HamiltonJacobiBuildConfig
{
   public:

      static void print() { std::cerr << "HamiltonJacobiBuildConfig" << std::endl; }
};

/****
 * Turn off support for float and long double.
 */
//template<> struct ConfigTagReal< HamiltonJacobiBuildConfig, float > { enum { enabled = false }; };
template<> struct ConfigTagReal< HamiltonJacobiBuildConfig, long double > { enum { enabled = false }; };

/****
 * Turn off support for short int and long int indexing.
 */
template<> struct ConfigTagIndex< HamiltonJacobiBuildConfig, short int >{ enum { enabled = false }; };
template<> struct ConfigTagIndex< HamiltonJacobiBuildConfig, long int >{ enum { enabled = false }; };

/****
 * Use of Grid is enabled for allowed dimensions and Real, Device and Index types.
 */
/*template< int Dimension, typename Real, typename Device, typename Index >
   struct ConfigTagMesh< HamiltonJacobiBuildConfig, Meshes::Grid< Dimension, Real, Device, Index > >
      { enum { enabled = ConfigTagDimension< HamiltonJacobiBuildConfig, Dimension >::enabled  &&
                         ConfigTagReal< HamiltonJacobiBuildConfig, Real >::enabled &&
                         ConfigTagDevice< HamiltonJacobiBuildConfig, Device >::enabled &&
                         ConfigTagIndex< HamiltonJacobiBuildConfig, Index >::enabled }; };*/

/****
 * Please, chose your preferred time discretisation  here.
 */
template<> struct ConfigTagTimeDiscretisation< HamiltonJacobiBuildConfig, ExplicitTimeDiscretisationTag >{ enum { enabled = true }; };
template<> struct ConfigTagTimeDiscretisation< HamiltonJacobiBuildConfig, SemiImplicitTimeDiscretisationTag >{ enum { enabled = true }; };
template<> struct ConfigTagTimeDiscretisation< HamiltonJacobiBuildConfig, ImplicitTimeDiscretisationTag >{ enum { enabled = false }; };

/****
 * Only the Runge-Kutta-Merson solver is enabled by default.
 */
//template<> struct ConfigTagExplicitSolver< HamiltonJacobiBuildConfig, ExplicitEulerSolverTag >{ enum { enabled = false }; };

//} // namespace Solvers
//} // namespace TNL
