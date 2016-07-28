/***************************************************************************
                          tnlIdentityOperator.h  -  description
                             -------------------
    begin                : Nov 17, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Functions/tnlFunctionAdapter.h>

namespace TNL {

template< typename Function >
void
tnlNeumannBoundaryConditionsBase< Function >::
configSetup( Config::ConfigDescription& config,
             const String& prefix )
{
   Function::configSetup( config, prefix );
}

template< typename Function >
bool
tnlNeumannBoundaryConditionsBase< Function >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   return this->function.setup( parameters, prefix );
}

template< typename Function >
void
tnlNeumannBoundaryConditionsBase< Function >::
setFunction( const Function& function )
{
   this->function = function;
}


template< typename Function >
Function&
tnlNeumannBoundaryConditionsBase< Function >::
getFunction()
{
   return this->function;
}

template< typename Function >
const Function&
tnlNeumannBoundaryConditionsBase< Function >::
getFunction() const
{
   return this->function;
}

/****
 * 1D grid
 */
/*
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
   template< typename EntityType,
             typename MeshFunction >
__cuda_callable__
const Real
tnlNeumannBoundaryConditions< tnlGrid< 1, MeshReal, Device, MeshIndex >, Function, Real, Index >::
operator()( const MeshFunction& u,
            const EntityType& entity,
            const RealType& time ) const
{
   const MeshType& mesh = entity.getMesh();
   auto neighbourEntities = entity.getNeighbourEntities();
   const IndexType& index = entity.getIndex();
   if( entity.getCoordinates().x() == 0 )
      return u[ neighbourEntities.template getEntityIndex< 1 >() ] - mesh.getSpaceSteps().x() * 
         tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
   else
      return u[ neighbourEntities.template getEntityIndex< -1 >() ] + mesh.getSpaceSteps().x() * 
         tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );   
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
tnlNeumannBoundaryConditions< tnlGrid< 1, MeshReal, Device, MeshIndex >, Function, Real, Index >::
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
             typename EntityType,
             typename MeshFunction >
__cuda_callable__
void
tnlNeumannBoundaryConditions< tnlGrid< 1, MeshReal, Device, MeshIndex >, Function, Real, Index >::
setMatrixElements( const RealType& time,
                    const MeshType& mesh,
                    const IndexType& index,
                    const EntityType& entity,
                    const MeshFunction& u,
                    DofVectorType& b,
                    Matrix& matrix ) const
{
   auto neighbourEntities = entity.getNeighbourEntities();
   typename Matrix::MatrixRow matrixRow = matrix.getRow( index );
   if( entity.getCoordinates().x() == 0 )
   {
      matrixRow.setElement( 0, index, 1.0 );
      matrixRow.setElement( 1, neighbourEntities.template getEntityIndex< 1 >(), -1.0 );
      b[ index ] = - mesh.getSpaceSteps().x() * 
         tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
   }
   else
   {
      matrixRow.setElement( 0, neighbourEntities.template getEntityIndex< -1 >(), -1.0 );
      matrixRow.setElement( 1, index, 1.0 );
      b[ index ] = mesh.getSpaceSteps().x() *
         tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
   }
}*/

/****
 * 2D grid
 */
/*
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
   template< typename EntityType,
             typename MeshFunction >
__cuda_callable__
const Real
tnlNeumannBoundaryConditions< tnlGrid< 2, MeshReal, Device, MeshIndex >, Function, Real, Index >::
operator()( const MeshFunction& u,
            const EntityType& entity,
            const RealType& time ) const
{
   const MeshType& mesh = entity.getMesh();
   auto neighbourEntities = entity.getNeighbourEntities();
   const IndexType& index = entity.getIndex();
   if( entity.getCoordinates().x() == 0 )
   {
      return u[ neighbourEntities.template getEntityIndex< 1, 0 >() ] - mesh.getSpaceSteps().x() *
         tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
   }
   if( entity.getCoordinates().x() == mesh.getDimensions().x() - 1 )
   {
      return u[ neighbourEntities.template getEntityIndex< -1, 0 >() ] + mesh.getSpaceSteps().x() *
         tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
   }
   if( entity.getCoordinates().y() == 0 )
   {
      return u[ neighbourEntities.template getEntityIndex< 0, 1 >() ] - mesh.getSpaceSteps().y() *
         tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
   }
   if( entity.getCoordinates().y() == mesh.getDimensions().y() - 1 )
   {
      return u[ neighbourEntities.template getEntityIndex< 0, -1 >() ] + mesh.getSpaceSteps().y() *
         tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
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
tnlNeumannBoundaryConditions< tnlGrid< 2, MeshReal, Device, MeshIndex >, Function, Real, Index >::
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
             typename EntityType,
             typename MeshFunction >
__cuda_callable__
void
tnlNeumannBoundaryConditions< tnlGrid< 2, MeshReal, Device, MeshIndex >, Function, Real, Index >::
setMatrixElements( const RealType& time,
                    const MeshType& mesh,
                    const IndexType& index,
                    const EntityType& entity,
                    const MeshFunction& u,
                    DofVectorType& b,
                    Matrix& matrix ) const
{
   auto neighbourEntities = entity.getNeighbourEntities();
   typename Matrix::MatrixRow matrixRow = matrix.getRow( index );
   if( entity.getCoordinates().x() == 0 )
   {
      matrixRow.setElement( 0, index,                                                1.0 );
      matrixRow.setElement( 1, neighbourEntities.template getEntityIndex< 1, 0 >(), -1.0 );
      b[ index ] = - mesh.getSpaceSteps().x() *
         tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
   }
   if( entity.getCoordinates().x() == mesh.getDimensions().x() - 1 )
   {
      matrixRow.setElement( 0, neighbourEntities.template getEntityIndex< -1, 0 >(), -1.0 );
      matrixRow.setElement( 1, index,                                                 1.0 );
      b[ index ] = mesh.getSpaceSteps().x() *
         tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
   }
   if( entity.getCoordinates().y() == 0 )
   {
      matrixRow.setElement( 0, index,                                                1.0 );
      matrixRow.setElement( 1, neighbourEntities.template getEntityIndex< 0, 1 >(), -1.0 );
      b[ index ] = - mesh.getSpaceSteps().y() *
         tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
   }
   if( entity.getCoordinates().y() == mesh.getDimensions().y() - 1 )
   {
      matrixRow.setElement( 0, neighbourEntities.template getEntityIndex< 0, -1 >(), -1.0 );
      matrixRow.setElement( 1, index,                                                 1.0 );
      b[ index ] = mesh.getSpaceSteps().y() *
         tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
   }
}*/

/****
 * 3D grid
 */
/*
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Function,
          typename Real,
          typename Index >
   template< typename EntityType,
             typename MeshFunction >
__cuda_callable__
const Real
tnlNeumannBoundaryConditions< tnlGrid< 3, MeshReal, Device, MeshIndex >, Function, Real, Index >::
operator()( const MeshFunction& u,
            const EntityType& entity,            
            const RealType& time ) const
{
   const MeshType& mesh = entity.getMesh();
   auto neighbourEntities = entity.getNeighbourEntities();
   const IndexType& index = entity.getIndex();
   if( entity.getCoordinates().x() == 0 )
   {
      return u[ neighbourEntities.template getEntityIndex< 1, 0, 0 >() ] - mesh.getSpaceSteps().x() *
         tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
   }
   if( entity.getCoordinates().x() == mesh.getDimensions().x() - 1 )
   {
      return u[ neighbourEntities.template getEntityIndex< -1, 0, 0 >() ] + mesh.getSpaceSteps().x() *
         tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
   }
   if( entity.getCoordinates().y() == 0 )
   {
      return u[ neighbourEntities.template getEntityIndex< 0, 1, 0 >() ] - mesh.getSpaceSteps().y() *
         tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
   }
   if( entity.getCoordinates().y() == mesh.getDimensions().y() - 1 )
   {
      return u[ neighbourEntities.template getEntityIndex< 0, -1, 0 >() ] + mesh.getSpaceSteps().y() *
         tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
   }
   if( entity.getCoordinates().z() == 0 )
   {
      return u[ neighbourEntities.template getEntityIndex< 0, 0, 1 >() ] - mesh.getSpaceSteps().z() *
         tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
   }
   if( entity.getCoordinates().z() == mesh.getDimensions().z() - 1 )
   {
      return u[ neighbourEntities.template getEntityIndex< 0, 0, -1 >() ] + mesh.getSpaceSteps().z() *
         tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
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
tnlNeumannBoundaryConditions< tnlGrid< 3, MeshReal, Device, MeshIndex >, Function, Real, Index >::
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
             typename EntityType,
             typename MeshFunction >
__cuda_callable__
void
tnlNeumannBoundaryConditions< tnlGrid< 3, MeshReal, Device, MeshIndex >, Function, Real, Index >::
setMatrixElements( const RealType& time,
                    const MeshType& mesh,
                    const IndexType& index,
                    const EntityType& entity,                    
                    const MeshFunction& u,
                    DofVectorType& b,
                    Matrix& matrix ) const
{
   auto neighbourEntities = entity.getNeighbourEntities();
   typename Matrix::MatrixRow matrixRow = matrix.getRow( index );
   if( entity.getCoordinates().x() == 0 )
   {
      matrixRow.setElement( 0, index,                                                   1.0 );
      matrixRow.setElement( 1, neighbourEntities.template getEntityIndex< 1, 0, 0 >(), -1.0 );
      b[ index ] = - mesh.getSpaceSteps().x() *
         tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
   }
   if( entity.getCoordinates().x() == mesh.getDimensions().x() - 1 )
   {
      matrixRow.setElement( 0, neighbourEntities.template getEntityIndex< -1, 0, 0 >(), -1.0 );
      matrixRow.setElement( 1, index,                                                    1.0 );
      b[ index ] = mesh.getSpaceSteps().x() *
         tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
   }
   if( entity.getCoordinates().y() == 0 )
   {
      matrixRow.setElement( 0, index,                                                   1.0 );
      matrixRow.setElement( 1, neighbourEntities.template getEntityIndex< 0, 1, 0 >(), -1.0 );
      b[ index ] = - mesh.getSpaceSteps().y() * 
         tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
   }
   if( entity.getCoordinates().y() == mesh.getDimensions().y() - 1 )
   {
      matrixRow.setElement( 0, neighbourEntities.template getEntityIndex< 0, -1, 0 >(), -1.0 );
      matrixRow.setElement( 1, index,                                                    1.0 );
      b[ index ] = mesh.getSpaceSteps().y() *
         tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
   }
   if( entity.getCoordinates().z() == 0 )
   {
      matrixRow.setElement( 0, index,                                                   1.0 );
      matrixRow.setElement( 1, neighbourEntities.template getEntityIndex< 0, 0, 1 >(), -1.0 );
      b[ index ] = - mesh.getSpaceSteps().z() *
         tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
   }
   if( entity.getCoordinates().z() == mesh.getDimensions().z() - 1 )
   {
      matrixRow.setElement( 0, neighbourEntities.template getEntityIndex< 0, 0, -1 >(), -1.0 );
      matrixRow.setElement( 1, index,                                                    1.0 );
      b[ index ] = mesh.getSpaceSteps().z() *
         tnlFunctionAdapter< MeshType, FunctionType >::getValue( this->function, entity, time );
   }
}
*/

} // namespace TNL

