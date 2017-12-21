/***************************************************************************
                          PDEProblem_impl.h  -  description
                             -------------------
    begin                : Jan 10, 2015
    copyright            : (C) 2015 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Problems {

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
String
PDEProblem< Mesh, Real, Device, Index >::
getTypeStatic()
{
   return String( "PDEProblem< " ) +
          Mesh :: getTypeStatic() + ", " +
          getType< Real >() + ", " +
          getType< Device >() + ", " +
          getType< Index >() + " >";
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
bool
PDEProblem< Mesh, Real, Device, Index >::
setMeshDependentData( const MeshPointer& mesh,
                      MeshDependentDataPointer& meshDependentData )
{
   /****
    * Set-up auxiliary data depending on the numerical mesh
    */
   return true;
}

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
void
PDEProblem< Mesh, Real, Device, Index >::
bindMeshDependentData( const MeshPointer& mesh,
                       MeshDependentDataPointer& meshDependentData )
{
}

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
bool
PDEProblem< Mesh, Real, Device, Index >::
preIterate( const RealType& time,
            const RealType& tau,
            const MeshPointer& meshPointer,
            DofVectorPointer& dofs,
            MeshDependentDataPointer& meshDependentData )
{
   return true;
}

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
void
PDEProblem< Mesh, Real, Device, Index >::
setExplicitBoundaryConditions( const RealType& time,
                               const MeshPointer& meshPointer,
                               DofVectorPointer& dofs,
                               MeshDependentDataPointer& meshDependentData )
{
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
             const MeshPointer& meshPointer,
             DofVectorPointer& dofs,
             MeshDependentDataPointer& meshDependentData )
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
