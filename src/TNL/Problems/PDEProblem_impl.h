/***************************************************************************
                          PDEProblem_impl.h  -  description
                             -------------------
    begin                : Jan 10, 2015
    copyright            : (C) 2015 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Problems/PDEProblem.h>
#include <TNL/Meshes/DistributedMeshes/SubdomainOverlapsGetter.h>

namespace TNL {
namespace Problems {

template< typename Mesh,
          typename Communicator,
          typename Real,
          typename Device,
          typename Index >
String
PDEProblem< Mesh, Communicator, Real, Device, Index >::
getType()
{
   return String( "PDEProblem< " ) +
          Mesh::getType() + ", " +
          TNL::getType< Real >() + ", " +
          TNL::getType< Device >() + ", " +
          TNL::getType< Index >() + " >";
}

template< typename Mesh,
          typename Communicator,
          typename Real,
          typename Device,
          typename Index >
String
PDEProblem< Mesh, Communicator, Real, Device, Index >::
getPrologHeader() const
{
   return String( "General PDE Problem" );
}

template< typename Mesh,
          typename Communicator,
          typename Real,
          typename Device,
          typename Index >
void
PDEProblem< Mesh, Communicator, Real, Device, Index >::
writeProlog( Logger& logger, const Config::ParameterContainer& parameters ) const
{
}

template< typename Mesh,
          typename Communicator,
          typename Real,
          typename Device,
          typename Index >
bool
PDEProblem< Mesh, Communicator, Real, Device, Index >::
writeEpilog( Logger& logger ) const
{
   return true;
}

template< typename Mesh,
          typename Communicator,
          typename Real,
          typename Device,
          typename Index >
typename PDEProblem< Mesh, Communicator, Real, Device, Index >::IndexType
PDEProblem< Mesh, Communicator, Real, Device, Index >::
subdomainOverlapSize()
{ 
   return 1;
}

template< typename Mesh,
          typename Communicator,
          typename Real,
          typename Device,
          typename Index >
void
PDEProblem< Mesh, Communicator, Real, Device, Index >::
getSubdomainOverlaps( const Config::ParameterContainer& parameters,
                      const String& prefix,
                      const MeshType& mesh,
                      SubdomainOverlapsType& lower,
                      SubdomainOverlapsType& upper )
{
   using namespace TNL::Meshes::DistributedMeshes;
   SubdomainOverlapsGetter< MeshType, CommunicatorType >::getOverlaps( mesh, lower, upper, this->subdomainOverlapSize() );
}
      


template< typename Mesh,
          typename Communicator,
          typename Real,
          typename Device,
          typename Index >
bool
PDEProblem< Mesh, Communicator, Real, Device, Index >::
setMeshDependentData( const MeshPointer& mesh,
                      MeshDependentDataPointer& meshDependentData )
{
   /****
    * Set-up auxiliary data depending on the numerical mesh
    */
   return true;
}

template< typename Mesh,
          typename Communicator,
          typename Real,
          typename Device,
          typename Index >
void
PDEProblem< Mesh, Communicator, Real, Device, Index >::
bindMeshDependentData( const MeshPointer& mesh,
                       MeshDependentDataPointer& meshDependentData )
{
}

template< typename Mesh,
          typename Communicator,
          typename Real,
          typename Device,
          typename Index >
bool
PDEProblem< Mesh, Communicator, Real, Device, Index >::
preIterate( const RealType& time,
            const RealType& tau,
            const MeshPointer& meshPointer,
            DofVectorPointer& dofs,
            MeshDependentDataPointer& meshDependentData )
{
   return true;
}

template< typename Mesh,
          typename Communicator,
          typename Real,
          typename Device,
          typename Index >
void
PDEProblem< Mesh, Communicator, Real, Device, Index >::
setExplicitBoundaryConditions( const RealType& time,
                               const MeshPointer& meshPointer,
                               DofVectorPointer& dofs,
                               MeshDependentDataPointer& meshDependentData )
{
}

template< typename Mesh,
          typename Communicator,
          typename Real,
          typename Device,
          typename Index >
    template< typename Matrix >
void
PDEProblem< Mesh, Communicator, Real, Device, Index >::
saveFailedLinearSystem( const Matrix& matrix,
                        const DofVectorType& dofs,
                        const DofVectorType& rhs ) const
{
    matrix.save( "failed-matrix.tnl" );
    dofs.save( "failed-dof.vec.tnl" );
    rhs.save( "failed-rhs.vec.tnl" );
    std::cerr << "The linear system has been saved to failed-{matrix,dof.vec,rhs.vec}.tnl" << std::endl;
}

template< typename Mesh,
          typename Communicator,
          typename Real,
          typename Device,
          typename Index >
bool
PDEProblem< Mesh, Communicator, Real, Device, Index >::
postIterate( const RealType& time,
             const RealType& tau,
             const MeshPointer& meshPointer,
             DofVectorPointer& dofs,
             MeshDependentDataPointer& meshDependentData )
{
   return true;
}

template< typename Mesh,
          typename Communicator,
          typename Real,
          typename Device,
          typename Index >
Solvers::SolverMonitor*
PDEProblem< Mesh, Communicator, Real, Device, Index >::
getSolverMonitor()
{
   return 0;
}

} // namespace Problems
} // namespace TNL
