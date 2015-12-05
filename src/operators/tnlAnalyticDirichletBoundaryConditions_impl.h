/***************************************************************************
           tnlAnalyticDirichletBoundaryConditions_impl.h  -  description
                             -------------------
    begin                : Nov 8, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef tnlAnalyticDirichletBoundaryConditions_IMPL_H
#define	tnlAnalyticDirichletBoundaryConditions_IMPL_H

template< int Dimensions,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
void
tnlAnalyticDirichletBoundaryConditions< tnlGrid< Dimensions, MeshReal, Device, MeshIndex >, Function, Real, Index >::
configSetup( tnlConfigDescription& config,
             const tnlString& prefix )
{
   Function::configSetup( config, prefix );
}

template< int Dimensions,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
bool
tnlAnalyticDirichletBoundaryConditions< tnlGrid< Dimensions, MeshReal, Device, MeshIndex >, Function, Real, Index >::
setup( const tnlParameterContainer& parameters,
       const tnlString& prefix )
{
   return function.setup( parameters, prefix );
}

template< int Dimensions,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
void
tnlAnalyticDirichletBoundaryConditions< tnlGrid< Dimensions, MeshReal, Device, MeshIndex >, Function, Real, Index >::
setFunction( const Function& function )
{
   this->function = function;
}

template< int Dimensions,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
Function&
tnlAnalyticDirichletBoundaryConditions< tnlGrid< Dimensions, MeshReal, Device, MeshIndex >, Function, Real, Index >::
getFunction()
{
   return this->function;
}

template< int Dimensions,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
const Function&
tnlAnalyticDirichletBoundaryConditions< tnlGrid< Dimensions, MeshReal, Device, MeshIndex >, Function, Real, Index >::
getFunction() const
{
   return this->function;
}

template< int Dimensions,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
   template< typename EntityType >
__cuda_callable__
void
tnlAnalyticDirichletBoundaryConditions< tnlGrid< Dimensions, MeshReal, Device, MeshIndex >, Function, Real, Index >::
setBoundaryConditions( const RealType& time,
                       const MeshType& mesh,
                       const IndexType index,
                       const EntityType& entity,
                       DofVectorType& u,
                       DofVectorType& fu ) const
{
   fu[ index ] = 0;
   u[ index ] = function.getValue( entity.getCenter(), time );
}

template< int Dimensions,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
   template< typename EntityType >
__cuda_callable__
Index
tnlAnalyticDirichletBoundaryConditions< tnlGrid< Dimensions, MeshReal, Device, MeshIndex >, Function, Real, Index >::
getLinearSystemRowLength( const MeshType& mesh,
                          const IndexType& index,
                          const EntityType& entity ) const
{
   return 1;
}

template< int Dimensions,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
   template< typename Matrix,
             typename EntityType >          
__cuda_callable__
void
tnlAnalyticDirichletBoundaryConditions< tnlGrid< Dimensions, MeshReal, Device, MeshIndex >, Function, Real, Index >::
updateLinearSystem( const RealType& time,
                    const MeshType& mesh,
                    const IndexType& index,
                    const EntityType& entity,
                    DofVectorType& u,
                    DofVectorType& b,
                    Matrix& matrix ) const
{
   typename Matrix::MatrixRow matrixRow = matrix.getRow( index );
   matrixRow.setElement( 0, index, 1.0 );
   b[ index ] = function.getValue( entity.getCenter(), time );
}

#endif	/* tnlAnalyticDirichletBoundaryConditions_IMPL_H */

