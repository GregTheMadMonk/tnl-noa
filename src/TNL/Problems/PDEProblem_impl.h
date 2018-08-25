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

namespace TNL {
namespace Problems {

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
String
PDEProblem< Mesh, Real, Device, Index >::
getType()
{
   return String( "PDEProblem< " ) +
          Mesh::getType() + ", " +
          TNL::getType< Real >() + ", " +
          Device::getDeviceType() + ", " +
          TNL::getType< Index >() + " >";
}

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
String
PDEProblem< Mesh, Real, Device, Index >::
getPrologHeader() const
{
   return String( "General PDE Problem" );
}

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
void
PDEProblem< Mesh, Real, Device, Index >::
writeProlog( Logger& logger, const Config::ParameterContainer& parameters ) const
{
}

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
bool
PDEProblem< Mesh, Real, Device, Index >::
writeEpilog( Logger& logger ) const
{
   return true;
}

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
void
PDEProblem< Mesh, Real, Device, Index >::
setMesh( MeshPointer& meshPointer)
{
   this->meshPointer = meshPointer;
}

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
const typename PDEProblem< Mesh, Real, Device, Index >::MeshPointer&
PDEProblem< Mesh, Real, Device, Index >::getMesh() const
{
   return this->meshPointer;
}

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
typename PDEProblem< Mesh, Real, Device, Index >::MeshPointer&
PDEProblem< Mesh, Real, Device, Index >::getMesh()
{
   return this->meshPointer;
}

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
void
PDEProblem< Mesh, Real, Device, Index >::
setCommonData( CommonDataPointer& commonData )
{
   this->commonDataPointer = commonData;
}

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
const typename PDEProblem< Mesh, Real, Device, Index >::CommonDataPointer&
PDEProblem< Mesh, Real, Device, Index >::
getCommonData() const
{
   return this->commonDataPointer;
}

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
typename PDEProblem< Mesh, Real, Device, Index >::CommonDataPointer&
PDEProblem< Mesh, Real, Device, Index >::
getCommonData()
{
   return this->commonDataPointer;
}

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
bool
PDEProblem< Mesh, Real, Device, Index >::
preIterate( const RealType& time,
            const RealType& tau,
            DofVectorPointer& dofs )
{
   return true;
}

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
    template< typename Matrix >
void
PDEProblem< Mesh, Real, Device, Index >::
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
          typename Real,
          typename Device,
          typename Index >
bool
PDEProblem< Mesh, Real, Device, Index >::
postIterate( const RealType& time,
             const RealType& tau,
             DofVectorPointer& dofs )
{
   return true;
}

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
Solvers::SolverMonitor*
PDEProblem< Mesh, Real, Device, Index >::
getSolverMonitor()
{
   return 0;
}

} // namespace Problems
} // namespace TNL
