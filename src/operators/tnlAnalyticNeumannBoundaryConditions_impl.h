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
   auto neighbourEntities = entity.getNeighbourEntities();
   fu[ index ] = 0;
   const Real functionValue = this->function.getValue( entity.getCenter(), time );
   if( entity.getCoordinates().x() == 0 )
      u[ index ] = u[ neighbourEntities.template getEntityIndex< 1 >() ] - mesh.getSpaceSteps().x() * functionValue;
   else
      u[ index ] = u[ neighbourEntities.template getEntityIndex< -1 >() ] + mesh.getSpaceSteps().x() * functionValue;
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
   auto neighbourEntities = entity.getNeighbourEntities();
   typename Matrix::MatrixRow matrixRow = matrix.getRow( index );
   const Real functionValue = this->function.getValue( entity.getCenter(), time );
   if( entity.getCoordinates().x() == 0 )
   {
      matrixRow.setElement( 0, index,                            1.0 );
      matrixRow.setElement( 1, neighbourEntities.template getEntityIndex< 1 >(), -1.0 );
      b[ index ] = - mesh.getSpaceSteps().x() * functionValue;
   }
   else
   {
      matrixRow.setElement( 0, neighbourEntities.template getEntityIndex< -1 >(), -1.0 );
      matrixRow.setElement( 1, index,                              1.0 );
      b[ index ] = mesh.getSpaceSteps().x() * functionValue;
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
   const Real functionValue = this->function.getValue( entity.getCenter(), time );
   if( entity.getCoordinates().x() == 0 )
   {
      u[ index ] = u[ neighbourEntities.template getEntityIndex< 1, 0 >() ] - mesh.getSpaceSteps().x() * functionValue;
      return;
   }
   if( entity.getCoordinates().x() == mesh.getDimensions().x() - 1 )
   {
      u[ index ] = u[ neighbourEntities.template getEntityIndex< -1, 0 >() ] + mesh.getSpaceSteps().x() * functionValue;
      return;
   }
   if( entity.getCoordinates().y() == 0 )
   {
      u[ index ] = u[ neighbourEntities.template getEntityIndex< 0, 1 >() ] - mesh.getSpaceSteps().y() * functionValue;
      return;
   }
   if( entity.getCoordinates().y() == mesh.getDimensions().y() - 1 )
   {
      u[ index ] = u[ neighbourEntities.template getEntityIndex< 0, -1 >() ] + mesh.getSpaceSteps().y() * functionValue;
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
   const Real functionValue = this->function.getValue( entity.getCenter(), time );
   auto neighbourEntities = entity.getNeighbourEntities();
   if( entity.getCoordinates().x() == 0 )
   {
      matrixRow.setElement( 0, index,                                                1.0 );
      matrixRow.setElement( 1, neighbourEntities.template getEntityIndex< 1, 0 >(), -1.0 );
      b[ index ] = - mesh.getSpaceSteps().x() * functionValue;
   }
   if( entity.getCoordinates().x() == mesh.getDimensions().x() - 1 )
   {
      matrixRow.setElement( 0, neighbourEntities.template getEntityIndex< -1, 0 >(), -1.0 );
      matrixRow.setElement( 1, index,                                                 1.0 );
      b[ index ] = mesh.getSpaceSteps().x() * functionValue;
   }
   if( entity.getCoordinates().y() == 0 )
   {
      matrixRow.setElement( 0, index,                                                1.0 );
      matrixRow.setElement( 1, neighbourEntities.template getEntityIndex< 0, 1 >(), -1.0 );
      b[ index ] = - mesh.getSpaceSteps().y() * functionValue;
   }
   if( entity.getCoordinates().y() == mesh.getDimensions().y() - 1 )
   {
      matrixRow.setElement( 0, neighbourEntities.template getEntityIndex< 0, -1 >(), -1.0 );
      matrixRow.setElement( 1, index,                                                 1.0 );
      b[ index ] = mesh.getSpaceSteps().y() * functionValue;
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
   auto neighbourEntities = entity.getNeighbourEntities();
   fu[ index ] = 0;
   fu[ index ] = 0;
   const Real functionValue = this->function.getValue( entity.getCenter(), time );
   if( entity.getCoordinates().x() == 0 )
   {
      u[ index ] = u[ neighbourEntities.template getEntityIndex< 1, 0, 0 >() ] - mesh.getSpaceSteps().x() * functionValue;
      return;
   }
   if( entity.getCoordinates().x() == mesh.getDimensions().x() - 1 )
   {
      u[ index ] = u[ neighbourEntities.template getEntityIndex< -1, 0, 0 >() ] + mesh.getSpaceSteps().x() * functionValue;
      return;
   }
   if( entity.getCoordinates().y() == 0 )
   {
      u[ index ] = u[ neighbourEntities.template getEntityIndex< 0, 1, 0 >() ] - mesh.getSpaceSteps().y() * functionValue;
      return;
   }
   if( entity.getCoordinates().y() == mesh.getDimensions().y() - 1 )
   {
      u[ index ] = u[ neighbourEntities.template getEntityIndex< 0, -1, 0 >() ] + mesh.getSpaceSteps().y() * functionValue;
      return;
   }
   if( entity.getCoordinates().z() == 0 )
   {
      u[ index ] = u[ neighbourEntities.template getEntityIndex< 0, 0, 1 >() ] - mesh.getSpaceSteps().z() * functionValue;
      return;
   }
   if( entity.getCoordinates().z() == mesh.getDimensions().z() - 1 )
   {
      u[ index ] = u[ neighbourEntities.template getEntityIndex< 0, 0, - 1 >() ] + mesh.getSpaceSteps().z() * functionValue;
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
   auto neighbourEntities = entity.getNeighbourEntities();
   typename Matrix::MatrixRow matrixRow = matrix.getRow( index );
   const Real functionValue = this->function.getValue( entity.getCenter(), time );
   if( entity.getCoordinates().x() == 0 )
   {
      matrixRow.setElement( 0, index,                                                   1.0 );
      matrixRow.setElement( 1, neighbourEntities.template getEntityIndex< 1, 0, 0 >(), -1.0 );
      b[ index ] = - mesh.getSpaceSteps().x() * functionValue;
   }
   if( entity.getCoordinates().x() == mesh.getDimensions().x() - 1 )
   {
      matrixRow.setElement( 0, neighbourEntities.template getEntityIndex< -1, 0, 0 >(), -1.0 );
      matrixRow.setElement( 1, index,                                                    1.0 );
      b[ index ] = mesh.getSpaceSteps().x() * functionValue;
   }
   if( entity.getCoordinates().y() == 0 )
   {
      matrixRow.setElement( 0, index,                                                   1.0 );
      matrixRow.setElement( 1, neighbourEntities.template getEntityIndex< 0, 1, 0 >(), -1.0 );
      b[ index ] = - mesh.getSpaceSteps().y() * functionValue;
   }
   if( entity.getCoordinates().y() == mesh.getDimensions().y() - 1 )
   {
      matrixRow.setElement( 0, neighbourEntities.template getEntityIndex< 0, -1, 0 >(), -1.0 );
      matrixRow.setElement( 1, index,                                                    1.0 );
      b[ index ] = mesh.getSpaceSteps().y() * functionValue;
   }
   if( entity.getCoordinates().z() == 0 )
   {
      matrixRow.setElement( 0, index,                                                   1.0 );
      matrixRow.setElement( 1, neighbourEntities.template getEntityIndex< 0, 0, 1 >(), -1.0 );
      b[ index ] = - mesh.getSpaceSteps().z() * functionValue;
   }
   if( entity.getCoordinates().z() == mesh.getDimensions().z() - 1 )
   {
      matrixRow.setElement( 0, neighbourEntities.template getEntityIndex< 0, 0, -1 >(), -1.0 );
      matrixRow.setElement( 1, index,                                                    1.0 );
      b[ index ] = mesh.getSpaceSteps().z() * functionValue;
   }
}


#endif /* TNLANALYTICNEUMANNBOUNDARYCONDITIONS_IMPL_H_ */
