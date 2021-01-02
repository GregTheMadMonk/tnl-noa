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
   using namespace Meshes::DistributedMeshes;
   SubdomainOverlapsGetter< MeshType >::getOverlaps( mesh.getDistributedMesh(), lower, upper, this->subdomainOverlapSize() );
}

template< typename Mesh,
          typename Communicator,
          typename Real,
          typename Device,
          typename Index >
void
PDEProblem< Mesh, Communicator, Real, Device, Index >::
setMesh( MeshPointer& meshPointer)
{
   this->meshPointer = meshPointer;
}

template< typename Mesh,
          typename Communicator,
          typename Real,
          typename Device,
          typename Index >
const typename PDEProblem< Mesh, Communicator, Real, Device, Index >::MeshPointer&
PDEProblem< Mesh, Communicator, Real, Device, Index >::getMesh() const
{
   return this->meshPointer;
}

template< typename Mesh,
          typename Communicator,
          typename Real,
          typename Device,
          typename Index >
typename PDEProblem< Mesh, Communicator, Real, Device, Index >::MeshPointer&
PDEProblem< Mesh, Communicator, Real, Device, Index >::getMesh()
{
   return this->meshPointer;
}

template< typename Mesh,
          typename Communicator,
          typename Real,
          typename Device,
          typename Index >
void
PDEProblem< Mesh, Communicator, Real, Device, Index >::
setCommonData( CommonDataPointer& commonData )
{
   this->commonDataPointer = commonData;
}

template< typename Mesh,
          typename Communicator,
          typename Real,
          typename Device,
          typename Index >
const typename PDEProblem< Mesh, Communicator, Real, Device, Index >::CommonDataPointer&
PDEProblem< Mesh, Communicator, Real, Device, Index >::
getCommonData() const
{
   return this->commonDataPointer;
}

template< typename Mesh,
          typename Communicator,
          typename Real,
          typename Device,
          typename Index >
typename PDEProblem< Mesh, Communicator, Real, Device, Index >::CommonDataPointer&
PDEProblem< Mesh, Communicator, Real, Device, Index >::
getCommonData()
{
   return this->commonDataPointer;
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
            DofVectorPointer& dofs )
{
   return true;
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
    File( "failed-dof.vec.tnl", std::ios_base::out ) << dofs;
    File( "failed-rhs.vec.tnl", std::ios_base::out ) << rhs;
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
             DofVectorPointer& dofs )
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
