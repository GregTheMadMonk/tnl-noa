/***************************************************************************
                          tnlPDEProblem_impl.h  -  description
                             -------------------
    begin                : Jan 10, 2015
    copyright            : (C) 2015 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
tnlString
tnlPDEProblem< Mesh, Real, Device, Index >::
getTypeStatic()
{
   return tnlString( "tnlPDEProblem< " ) +
          Mesh :: getTypeStatic() + ", " +
          getType< Real >() + ", " +
          getType< Device >() + ", " +
          getType< Index >() + " >";
}

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
tnlString
tnlPDEProblem< Mesh, Real, Device, Index >::
getPrologHeader() const
{
   return tnlString( "General PDE Problem" );
}

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
void
tnlPDEProblem< Mesh, Real, Device, Index >::
writeProlog( tnlLogger& logger, const tnlParameterContainer& parameters ) const
{
}

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
bool
tnlPDEProblem< Mesh, Real, Device, Index >::
writeEpilog( tnlLogger& logger ) const
{
   return true;
}


template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
bool
tnlPDEProblem< Mesh, Real, Device, Index >::
setMeshDependentData( const MeshType& mesh,
                      MeshDependentDataType& meshDependentData )
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
tnlPDEProblem< Mesh, Real, Device, Index >::
bindMeshDependentData( const MeshType& mesh,
                       MeshDependentDataType& meshDependentData )
{
}

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
bool
tnlPDEProblem< Mesh, Real, Device, Index >::
preIterate( const RealType& time,
            const RealType& tau,
            const MeshType& mesh,
            DofVectorType& dofs,
            DofVectorType& auxDofs )
{
   return true;
}

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
void
tnlPDEProblem< Mesh, Real, Device, Index >::
setExplicitBoundaryConditions( const RealType& time,
                               const MeshType& mesh,
                               DofVectorType& dofs,
                               MeshDependentDataType& meshDependentData )
{
}


template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
bool
tnlPDEProblem< Mesh, Real, Device, Index >::
postIterate( const RealType& time,
             const RealType& tau,
             const MeshType& mesh,
             DofVectorType& dofs,
             DofVectorType& auxDofs )
{
   return true;
}

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
tnlSolverMonitor< typename tnlPDEProblem< Mesh, Real, Device, Index >::RealType,
                  typename tnlPDEProblem< Mesh, Real, Device, Index >::IndexType >*
tnlPDEProblem< Mesh, Real, Device, Index >::
getSolverMonitor()
{
   return 0;
}

} // namespace TNL
