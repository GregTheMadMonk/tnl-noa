/***************************************************************************
                          tnlAnalyticNeumannBoundaryConditions_impl.h  -  description
                             -------------------
    begin                : Nov 22, 2014
    copyright            : (C) 2014 by oberhuber
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

#ifndef TNLANALYTICNEUMANNBOUNDARYCONDITIONS_IMPL_H_
#define TNLANALYTICNEUMANNBOUNDARYCONDITIONS_IMPL_H_

/****
 * Base
 */
template< typename Function >
void
tnlAnalyticNeumannBoundaryConditionsBase< Function >::
configSetup( tnlConfigDescription& config,
             const tnlString& prefix )
{
   Function::configSetup( config, prefix );
}

template< typename Function >
bool
tnlAnalyticNeumannBoundaryConditionsBase< Function >::
setup( const tnlParameterContainer& parameters,
       const tnlString& prefix )
{
   return function.setup( parameters, prefix );
}

template< typename Function >
void
tnlAnalyticNeumannBoundaryConditionsBase< Function >::
setFunction( const Function& function )
{
   this->function = function;
}

template< typename Function >
Function&
tnlAnalyticNeumannBoundaryConditionsBase< Function >::
getFunction()
{
   return this->function;
}

template< typename Function >
const Function&
tnlAnalyticNeumannBoundaryConditionsBase< Function >::
getFunction() const
{
   return this->function;
}

/****
 * 1D grid
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
   template< typename EntityType >
__cuda_callable__
void
tnlAnalyticNeumannBoundaryConditions< tnlGrid< 1, MeshReal, Device, MeshIndex >, Function, Real, Index >::
setBoundaryConditions( const RealType& time,
                       const MeshType& mesh,
                       const IndexType index,
                       const EntityType& entity,
                       DofVectorType& u,
                       DofVectorType& fu ) const
{
   fu[ index ] = 0;
   const Real functionValue = this->function.getValue( mesh.getEntityCenter( entity ), time );
   if( entity.getCoordinates().x() == 0 )
      u[ index ] = u[ mesh.template getCellNextToCell< 1 >( index ) ] - mesh.getHx() * functionValue;
   else
      u[ index ] = u[ mesh.template getCellNextToCell< -1 >( index ) ] + mesh.getHx() * functionValue;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
   template< typename EntityType >
__cuda_callable__
Index
tnlAnalyticNeumannBoundaryConditions< tnlGrid< 1, MeshReal, Device, MeshIndex >, Function, Real, Index >::
getLinearSystemRowLength( const MeshType& mesh,
                          const IndexType& index,
                          const EntityType& entity ) const
{
   return 2;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
   template< typename Matrix,
             typename EntityType >
__cuda_callable__
void
tnlAnalyticNeumannBoundaryConditions< tnlGrid< 1, MeshReal, Device, MeshIndex >, Function, Real, Index >::
updateLinearSystem( const RealType& time,
                    const MeshType& mesh,
                    const IndexType& index,
                    const EntityType& entity,
                    DofVectorType& u,
                    DofVectorType& b,
                    Matrix& matrix ) const
{
   typename Matrix::MatrixRow matrixRow = matrix.getRow( index );
   const Real functionValue = this->function.getValue( mesh.getEntityCenter( entity ), time );
   if( entity.getCoordinates().x() == 0 )
   {
      matrixRow.setElement( 0, index,                            1.0 );
      matrixRow.setElement( 1, mesh.template getCellNextToCell< 1 >( index ), -1.0 );
      b[ index ] = - mesh.getHx() * functionValue;
   }
   else
   {
      matrixRow.setElement( 0, mesh.template getCellNextToCell< -1 >( index ), -1.0 );
      matrixRow.setElement( 1, index,                              1.0 );
      b[ index ] = mesh.getHx() * functionValue;
   }
}

/****
 * 2D grid
 */

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
   template< typename EntityType >          
__cuda_callable__
void
tnlAnalyticNeumannBoundaryConditions< tnlGrid< 2, MeshReal, Device, MeshIndex >, Function, Real, Index >::
setBoundaryConditions( const RealType& time,
                       const MeshType& mesh,
                       const IndexType index,
                       const EntityType& entity,
                       DofVectorType& u,
                       DofVectorType& fu ) const
{
   auto neighbourEntities = entity.getNeighbourEntities();
   fu[ index ] = 0;
   const Real functionValue = this->function.getValue( mesh.getEntityCenter( entity ), time );
   if( entity.getCoordinates().x() == 0 )
   {
      u[ index ] = u[ neighbourEntities.template getEntityIndex< 1, 0 >() ] - mesh.getHx() * functionValue;
      return;
   }
   if( entity.getCoordinates().x() == mesh.getDimensions().x() - 1 )
   {
      u[ index ] = u[ neighbourEntities.template getEntityIndex< -1, 0 >() ] + mesh.getHx() * functionValue;
      return;
   }
   if( entity.getCoordinates().y() == 0 )
   {
      u[ index ] = u[ neighbourEntities.template getEntityIndex< 0, 1 >() ] - mesh.getHy() * functionValue;
      return;
   }
   if( entity.getCoordinates().y() == mesh.getDimensions().y() - 1 )
   {
      u[ index ] = u[ neighbourEntities.template getEntityIndex< 0, -1 >() ] + mesh.getHy() * functionValue;
      return;
   }
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
   template< typename EntityType >          
__cuda_callable__
Index
tnlAnalyticNeumannBoundaryConditions< tnlGrid< 2, MeshReal, Device, MeshIndex >, Function, Real, Index >::
getLinearSystemRowLength( const MeshType& mesh,
                          const IndexType& index,
                          const EntityType& entity ) const
{
   return 2;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
   template< typename Matrix,
             typename EntityType >
__cuda_callable__
void
tnlAnalyticNeumannBoundaryConditions< tnlGrid< 2, MeshReal, Device, MeshIndex >, Function, Real, Index >::
updateLinearSystem( const RealType& time,
                    const MeshType& mesh,
                    const IndexType& index,
                    const EntityType& entity,
                    DofVectorType& u,
                    DofVectorType& b,
                    Matrix& matrix ) const
{
   typename Matrix::MatrixRow matrixRow = matrix.getRow( index );
   const Real functionValue = this->function.getValue( mesh.getEntityCenter( entity ), time );
   auto neighbourEntities = entity.getNeighbourEntities();
   if( entity.getCoordinates().x() == 0 )
   {
      matrixRow.setElement( 0, index,                                                1.0 );
      matrixRow.setElement( 1, neighbourEntities.template getEntityIndex< 1, 0 >(), -1.0 );
      b[ index ] = - mesh.getHx() * functionValue;
   }
   if( entity.getCoordinates().x() == mesh.getDimensions().x() - 1 )
   {
      matrixRow.setElement( 0, neighbourEntities.template getEntityIndex< -1, 0 >(), -1.0 );
      matrixRow.setElement( 1, index,                                                 1.0 );
      b[ index ] = mesh.getHx() * functionValue;
   }
   if( entity.getCoordinates().y() == 0 )
   {
      matrixRow.setElement( 0, index,                                                1.0 );
      matrixRow.setElement( 1, neighbourEntities.template getEntityIndex< 0, 1 >(), -1.0 );
      b[ index ] = - mesh.getHy() * functionValue;
   }
   if( entity.getCoordinates().y() == mesh.getDimensions().y() - 1 )
   {
      matrixRow.setElement( 0, neighbourEntities.template getEntityIndex< 0, -1 >(), -1.0 );
      matrixRow.setElement( 1, index,                                                 1.0 );
      b[ index ] = mesh.getHy() * functionValue;
   }
}

/****
 * 3D grid
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
   template< typename EntityType >          
__cuda_callable__
void
tnlAnalyticNeumannBoundaryConditions< tnlGrid< 3, MeshReal, Device, MeshIndex >, Function, Real, Index >::
setBoundaryConditions( const RealType& time,
                       const MeshType& mesh,
                       const IndexType index,
                       const EntityType& entity,
                       DofVectorType& u,
                       DofVectorType& fu ) const
{
   fu[ index ] = 0;
   fu[ index ] = 0;
   const Real functionValue = this->function.getValue( mesh.getEntityCenter( entity ), time );
   if( entity.getCoordinates().x() == 0 )
   {
      u[ index ] = u[ mesh.template getCellNextToCell< 1, 0, 0 >( index ) ] - mesh.getHx() * functionValue;
      return;
   }
   if( entity.getCoordinates().x() == mesh.getDimensions().x() - 1 )
   {
      u[ index ] = u[ mesh.template getCellNextToCell< -1, 0, 0 >( index ) ] + mesh.getHx() * functionValue;
      return;
   }
   if( entity.getCoordinates().y() == 0 )
   {
      u[ index ] = u[ mesh.template getCellNextToCell< 0, 1, 0 >( index ) ] - mesh.getHy() * functionValue;
      return;
   }
   if( entity.getCoordinates().y() == mesh.getDimensions().y() - 1 )
   {
      u[ index ] = u[ mesh.template getCellNextToCell< 0, -1, 0 >( index ) ] + mesh.getHy() * functionValue;
      return;
   }
   if( entity.getCoordinates().z() == 0 )
   {
      u[ index ] = u[ mesh.template getCellNextToCell< 0, 0, 1 >( index ) ] - mesh.getHz() * functionValue;
      return;
   }
   if( entity.getCoordinates().z() == mesh.getDimensions().z() - 1 )
   {
      u[ index ] = u[ mesh.template getCellNextToCell< 0, 0, - 1 >( index ) ] + mesh.getHz() * functionValue;
      return;
   }
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
   template< typename EntityType >          
__cuda_callable__
Index
tnlAnalyticNeumannBoundaryConditions< tnlGrid< 3, MeshReal, Device, MeshIndex >, Function, Real, Index >::
getLinearSystemRowLength( const MeshType& mesh,
                          const IndexType& index,
                          const EntityType& entity ) const
{
   return 2;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
   template< typename Matrix,
             typename EntityType >
__cuda_callable__
void
tnlAnalyticNeumannBoundaryConditions< tnlGrid< 3, MeshReal, Device, MeshIndex >, Function, Real, Index >::
updateLinearSystem( const RealType& time,
                    const MeshType& mesh,
                    const IndexType& index,
                    const EntityType& entity,
                    DofVectorType& u,
                    DofVectorType& b,
                    Matrix& matrix ) const
{
   typename Matrix::MatrixRow matrixRow = matrix.getRow( index );
   const Real functionValue = this->function.getValue( mesh.getEntityCenter( entity ), time );
   if( entity.getCoordinates().x() == 0 )
   {
      matrixRow.setElement( 0, index,                            1.0 );
      matrixRow.setElement( 1, mesh.template getCellNextToCell< 1, 0, 0 >( index ), -1.0 );
      b[ index ] = - mesh.getHx() * functionValue;
   }
   if( entity.getCoordinates().x() == mesh.getDimensions().x() - 1 )
   {
      matrixRow.setElement( 0, mesh.template getCellNextToCell< -1, 0, 0 >( index ), -1.0 );
      matrixRow.setElement( 1, index,                              1.0 );
      b[ index ] = mesh.getHx() * functionValue;
   }
   if( entity.getCoordinates().y() == 0 )
   {
      matrixRow.setElement( 0, index,                            1.0 );
      matrixRow.setElement( 1, mesh.template getCellNextToCell< 0, 1, 0 >( index ), -1.0 );
      b[ index ] = - mesh.getHy() * functionValue;
   }
   if( entity.getCoordinates().y() == mesh.getDimensions().y() - 1 )
   {
      matrixRow.setElement( 0, mesh.template getCellNextToCell< 0, -1, 0 >( index ), -1.0 );
      matrixRow.setElement( 1, index,                              1.0 );
      b[ index ] = mesh.getHy() * functionValue;
   }
   if( entity.getCoordinates().z() == 0 )
   {
      matrixRow.setElement( 0, index,                           1.0 );
      matrixRow.setElement( 1, mesh.template getCellNextToCell< 0, 0, 1 >( index ), -1.0 );
      b[ index ] = - mesh.getHz() * functionValue;
   }
   if( entity.getCoordinates().z() == mesh.getDimensions().z() - 1 )
   {
      matrixRow.setElement( 0, mesh.template getCellNextToCell< 0, 0, -1 >( index ), -1.0 );
      matrixRow.setElement( 1, index,                              1.0 );
      b[ index ] = mesh.getHz() * functionValue;
   }
}


#endif /* TNLANALYTICNEUMANNBOUNDARYCONDITIONS_IMPL_H_ */
