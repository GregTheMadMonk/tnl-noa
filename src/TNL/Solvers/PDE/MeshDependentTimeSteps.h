/***************************************************************************
                          MeshDependentTimeSteps.h  -  description
                             -------------------
    begin                : Jan 26, 2017
    copyright            : (C) 2017 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/Mesh.h>

namespace TNL {
namespace Solvers {
namespace PDE {   

template< typename Mesh, typename Real >
class MeshDependentTimeSteps
{
};

template< int Dimension,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
class MeshDependentTimeSteps< TNL::Meshes::Grid< Dimension, MeshReal, Device, MeshIndex >, Real >
{
public:
   using MeshType = TNL::Meshes::Grid< Dimension, MeshReal, Device, MeshIndex >;

   bool setTimeStepOrder( const Real& timeStepOrder )
   {
      if( timeStepOrder < 0 ) {
         std::cerr << "The time step order for PDESolver must be zero or positive value." << std::endl;
         return false;
      }
      this->timeStepOrder = timeStepOrder;
      return true;
   }

   const Real& getTimeStepOrder() const
   {
      return timeStepOrder;
   }

   Real getRefinedTimeStep( const MeshType& mesh, const Real& timeStep )
   {
      return timeStep * std::pow( mesh.getSmallestSpaceStep(), this->timeStepOrder );
   }

protected:
   Real timeStepOrder = 0.0;
};

template< typename MeshConfig,
          typename Device,
          typename Real >
class MeshDependentTimeSteps< TNL::Meshes::Mesh< MeshConfig, Device >, Real >
{
public:
   using MeshType = TNL::Meshes::Mesh< MeshConfig >;

   bool setTimeStepOrder( const Real& timeStepOrder )
   {
      if( timeStepOrder != 0.0 ) {
         std::cerr << "Mesh-dependent time stepping is not available on unstructured meshes, so the time step order must be 0." << std::endl;
         return false;
      }
      this->timeStepOrder = timeStepOrder;
      return true;
   }

   const Real& getTimeStepOrder() const
   {
      return timeStepOrder;
   }

   Real getRefinedTimeStep( const MeshType& mesh, const Real& timeStep )
   {
      return timeStep;
   }

protected:
   Real timeStepOrder = 0.0;
};

} // namespace PDE
} // namespace Solvers
} // namespace TNL
