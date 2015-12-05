#ifndef TNLNEUMANNBOUNDARYCONDITIONS_IMPL_H
#define	TNLNEUMANNBOUNDARYCONDITIONS_IMPL_H

template< typename Vector >
void
tnlNeumannBoundaryConditionsBase< Vector >::
configSetup( tnlConfigDescription& config,
             const tnlString& prefix )
{
   config.addEntry     < tnlString >( prefix + "file", "Data for the boundary conditions." );
}

template< typename Vector >
bool
tnlNeumannBoundaryConditionsBase< Vector >::
setup( const tnlParameterContainer& parameters,
       const tnlString& prefix )
{
   if( parameters.checkParameter( prefix + "file" ) )
   {
      tnlString fileName = parameters.getParameter< tnlString >( prefix + "file" );
      if( ! this->vector.load( fileName ) )
         return false;
   }
   return true;
}

template< typename Vector >
Vector&
tnlNeumannBoundaryConditionsBase< Vector >::
getVector()
{
   return this->vector;
}

template< typename Vector >
const Vector&
tnlNeumannBoundaryConditionsBase< Vector >::
getVector() const
{
   return this->vector;
}

/****
 * 1D grid
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Vector,
          typename Real,
          typename Index >
   template< typename EntityType >          
__cuda_callable__
void
tnlNeumannBoundaryConditions< tnlGrid< 1, MeshReal, Device, MeshIndex >, Vector, Real, Index >::
setBoundaryConditions( const RealType& time,
                       const MeshType& mesh,
                       const IndexType index,
                       const EntityType& entity,
                       DofVectorType& u,
                       DofVectorType& fu ) const
{
   auto neighbourEntities = entity.getNeighbourEntities();
   fu[ index ] = 0;
   if( entity.getCoordinates().x() == 0 )
      u[ index ] = u[ neighbourEntities.template getEntityIndex< 1 >() ] - mesh.getSpaceSteps().x() * this->vector[ index ];
   else
      u[ index ] = u[ neighbourEntities.template getEntityIndex< -1 >() ] + mesh.getSpaceSteps().x() * this->vector[ index ];
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Vector,
          typename Real,
          typename Index >
   template< typename EntityType >
__cuda_callable__
Index
tnlNeumannBoundaryConditions< tnlGrid< 1, MeshReal, Device, MeshIndex >, Vector, Real, Index >::
getLinearSystemRowLength( const MeshType& mesh,
                          const IndexType& index,
                          const EntityType& entity ) const
{
   return 2;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Vector,
          typename Real,
          typename Index >
   template< typename Matrix,
             typename EntityType >
__cuda_callable__
void
tnlNeumannBoundaryConditions< tnlGrid< 1, MeshReal, Device, MeshIndex >, Vector, Real, Index >::
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
   if( entity.getCoordinates().x() == 0 )
   {
      matrixRow.setElement( 0, index, 1.0 );
      matrixRow.setElement( 1, neighbourEntities.template getEntityIndex< 1 >(), -1.0 );
      b[ index ] = - mesh.getSpaceSteps().x() * this->vector[ index];
   }
   else
   {
      matrixRow.setElement( 0, neighbourEntities.template getEntityIndex< -1 >(), -1.0 );
      matrixRow.setElement( 1, index, 1.0 );
      b[ index ] = mesh.getSpaceSteps().x() * this->vector[ index ];
   }
}

/****
 * 2D grid
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Vector,
          typename Real,
          typename Index >
   template< typename EntityType >          
__cuda_callable__
void
tnlNeumannBoundaryConditions< tnlGrid< 2, MeshReal, Device, MeshIndex >, Vector, Real, Index >::
setBoundaryConditions( const RealType& time,
                       const MeshType& mesh,
                       const IndexType index,
                       const EntityType& entity,
                       DofVectorType& u,
                       DofVectorType& fu ) const
{
   auto neighbourEntities = entity.getNeighbourEntities();
   fu[ index ] = 0;
   if( entity.getCoordinates().x() == 0 )
   {
      u[ index ] = u[ neighbourEntities.template getEntityIndex< 1, 0 >() ] - mesh.getSpaceSteps().x() * this->vector[ index ];
      return;
   }
   if( entity.getCoordinates().x() == mesh.getDimensions().x() - 1 )
   {
      u[ index ] = u[ neighbourEntities.template getEntityIndex< -1, 0 >() ] + mesh.getSpaceSteps().x() * this->vector[ index ];
      return;
   }
   if( entity.getCoordinates().y() == 0 )
   {
      u[ index ] = u[ neighbourEntities.template getEntityIndex< 0, 1 >() ] - mesh.getSpaceSteps().y() * this->vector[ index ];
      return;
   }
   if( entity.getCoordinates().y() == mesh.getDimensions().y() - 1 )
   {
      u[ index ] = u[ neighbourEntities.template getEntityIndex< 0, -1 >() ] + mesh.getSpaceSteps().y() * this->vector[ index ];
      return;
   }
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Vector,
          typename Real,
          typename Index >
   template< typename EntityType >          
__cuda_callable__
Index
tnlNeumannBoundaryConditions< tnlGrid< 2, MeshReal, Device, MeshIndex >, Vector, Real, Index >::
getLinearSystemRowLength( const MeshType& mesh,
                          const IndexType& index,
                          const EntityType& entity ) const
{
   return 2;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Vector,
          typename Real,
          typename Index >
   template< typename Matrix,
             typename EntityType >
__cuda_callable__
void
tnlNeumannBoundaryConditions< tnlGrid< 2, MeshReal, Device, MeshIndex >, Vector, Real, Index >::
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
   if( entity.getCoordinates().x() == 0 )
   {
      matrixRow.setElement( 0, index,                                                1.0 );
      matrixRow.setElement( 1, neighbourEntities.template getEntityIndex< 1, 0 >(), -1.0 );
      b[ index ] = - mesh.getSpaceSteps().x() * this->vector[ index ];
   }
   if( entity.getCoordinates().x() == mesh.getDimensions().x() - 1 )
   {
      matrixRow.setElement( 0, neighbourEntities.template getEntityIndex< -1, 0 >(), -1.0 );
      matrixRow.setElement( 1, index,                                                 1.0 );
      b[ index ] = mesh.getSpaceSteps().x() * this->vector[ index ];
   }
   if( entity.getCoordinates().y() == 0 )
   {
      matrixRow.setElement( 0, index,                                                1.0 );
      matrixRow.setElement( 1, neighbourEntities.template getEntityIndex< 0, 1 >(), -1.0 );
      b[ index ] = - mesh.getSpaceSteps().y() * this->vector[ index ];
   }
   if( entity.getCoordinates().y() == mesh.getDimensions().y() - 1 )
   {
      matrixRow.setElement( 0, neighbourEntities.template getEntityIndex< 0, -1 >(), -1.0 );
      matrixRow.setElement( 1, index,                                                 1.0 );
      b[ index ] = mesh.getSpaceSteps().y() * this->vector[ index ];
   }
}

/****
 * 3D grid
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Vector,
          typename Real,
          typename Index >
   template< typename EntityType >          
__cuda_callable__
void
tnlNeumannBoundaryConditions< tnlGrid< 3, MeshReal, Device, MeshIndex >, Vector, Real, Index >::
setBoundaryConditions( const RealType& time,
                       const MeshType& mesh,
                       const IndexType index,
                       const EntityType& entity,
                       DofVectorType& u,
                       DofVectorType& fu ) const
{
   auto neighbourEntities = entity.getNeighbourEntities();
   fu[ index ] = 0;
   if( entity.getCoordinates().x() == 0 )
   {
      u[ index ] = u[ neighbourEntities.template getEntityIndex< 1, 0, 0 >() ] - mesh.getSpaceSteps().x() * this->vector[ index ];
      return;
   }
   if( entity.getCoordinates().x() == mesh.getDimensions().x() - 1 )
   {
      u[ index ] = u[ neighbourEntities.template getEntityIndex< -1, 0, 0 >() ] + mesh.getSpaceSteps().x() * this->vector[ index ];
      return;
   }
   if( entity.getCoordinates().y() == 0 )
   {
      u[ index ] = u[ neighbourEntities.template getEntityIndex< 0, 1, 0 >() ] - mesh.getSpaceSteps().y() * this->vector[ index ];
      return;
   }
   if( entity.getCoordinates().y() == mesh.getDimensions().y() - 1 )
   {
      u[ index ] = u[ neighbourEntities.template getEntityIndex< 0, -1, 0 >() ] + mesh.getSpaceSteps().y() * this->vector[ index ];
      return;
   }
   if( entity.getCoordinates().z() == 0 )
   {
      u[ index ] = u[ neighbourEntities.template getEntityIndex< 0, 0, 1 >() ] - mesh.getSpaceSteps().z() * this->vector[ index ];
      return;
   }
   if( entity.getCoordinates().z() == mesh.getDimensions().z() - 1 )
   {
      u[ index ] = u[ neighbourEntities.template getEntityIndex< 0, 0, -1 >() ] + mesh.getSpaceSteps().z() * this->vector[ index ];
      return;
   }
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Vector,
          typename Real,
          typename Index >
   template< typename EntityType >          
__cuda_callable__
Index
tnlNeumannBoundaryConditions< tnlGrid< 3, MeshReal, Device, MeshIndex >, Vector, Real, Index >::
getLinearSystemRowLength( const MeshType& mesh,
                          const IndexType& index,
                          const EntityType& entity ) const
{
   return 2;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Vector,
          typename Real,
          typename Index >
   template< typename Matrix,
             typename EntityType >
__cuda_callable__
void
tnlNeumannBoundaryConditions< tnlGrid< 3, MeshReal, Device, MeshIndex >, Vector, Real, Index >::
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
   if( entity.getCoordinates().x() == 0 )
   {
      matrixRow.setElement( 0, index,                                                   1.0 );
      matrixRow.setElement( 1, neighbourEntities.template getEntityIndex< 1, 0, 0 >(), -1.0 );
      b[ index ] = - mesh.getSpaceSteps().x() * this->vector[ index ];
   }
   if( entity.getCoordinates().x() == mesh.getDimensions().x() - 1 )
   {
      matrixRow.setElement( 0, neighbourEntities.template getEntityIndex< -1, 0, 0 >(), -1.0 );
      matrixRow.setElement( 1, index,                                                    1.0 );
      b[ index ] = mesh.getSpaceSteps().x() * this->vector[ index ];
   }
   if( entity.getCoordinates().y() == 0 )
   {
      matrixRow.setElement( 0, index,                                                   1.0 );
      matrixRow.setElement( 1, neighbourEntities.template getEntityIndex< 0, 1, 0 >(), -1.0 );
      b[ index ] = - mesh.getSpaceSteps().y() * this->vector[ index ];
   }
   if( entity.getCoordinates().y() == mesh.getDimensions().y() - 1 )
   {
      matrixRow.setElement( 0, neighbourEntities.template getEntityIndex< 0, -1, 0 >(), -1.0 );
      matrixRow.setElement( 1, index,                                                    1.0 );
      b[ index ] = mesh.getSpaceSteps().y() * this->vector[ index ];
   }
   if( entity.getCoordinates().z() == 0 )
   {
      matrixRow.setElement( 0, index,                                                   1.0 );
      matrixRow.setElement( 1, neighbourEntities.template getEntityIndex< 0, 0, 1 >(), -1.0 );
      b[ index ] = - mesh.getSpaceSteps().z() * this->vector[ index ];
   }
   if( entity.getCoordinates().z() == mesh.getDimensions().z() - 1 )
   {
      matrixRow.setElement( 0, neighbourEntities.template getEntityIndex< 0, 0, -1 >(), -1.0 );
      matrixRow.setElement( 1, index,                                                    1.0 );
      b[ index ] = mesh.getSpaceSteps().z() * this->vector[ index ];
   }
}

#endif	/* TNLNEUMANNBOUNDARYCONDITIONS_IMPL_H */

