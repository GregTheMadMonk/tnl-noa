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
__cuda_callable__
void
tnlNeumannBoundaryConditions< tnlGrid< 1, MeshReal, Device, MeshIndex >, Vector, Real, Index >::
setBoundaryConditions( const RealType& time,
                       const MeshType& mesh,
                       const IndexType index,
                       const CoordinatesType& coordinates,
                       DofVectorType& u,
                       DofVectorType& fu ) const
{
   fu[ index ] = 0;
   if( coordinates.x() == 0 )
      u[ index ] = u[ mesh.template getCellNextToCell< 1 >( index ) ] - mesh.getHx() * this->vector[ index ];
   else
      u[ index ] = u[ mesh.template getCellNextToCell< -1 >( index ) ] + mesh.getHx() * this->vector[ index ];
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Vector,
          typename Real,
          typename Index >
__cuda_callable__
Index
tnlNeumannBoundaryConditions< tnlGrid< 1, MeshReal, Device, MeshIndex >, Vector, Real, Index >::
getLinearSystemRowLength( const MeshType& mesh,
                          const IndexType& index,
                          const CoordinatesType& coordinates ) const
{
   return 2;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Vector,
          typename Real,
          typename Index >
   template< typename MatrixRow >
__cuda_callable__
void
tnlNeumannBoundaryConditions< tnlGrid< 1, MeshReal, Device, MeshIndex >, Vector, Real, Index >::
updateLinearSystem( const RealType& time,
                    const MeshType& mesh,
                    const IndexType& index,
                    const CoordinatesType& coordinates,
                    DofVectorType& u,
                    DofVectorType& b,
                    MatrixRow& matrixRow ) const
{
   if( coordinates.x() == 0 )
   {
      matrixRow.setElement( 0, index, 1.0 );
      matrixRow.setElement( 1, mesh.template getCellNextToCell< 1 >( index ), -1.0 );
      b[ index ] = - mesh.getHx() * this->vector[ index];
   }
   else
   {
      matrixRow.setElement( 0, mesh.template getCellNextToCell< -1 >( index ), -1.0 );
      matrixRow.setElement( 1, index, 1.0 );
      b[ index ] = mesh.getHx() * this->vector[ index ];
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
__cuda_callable__
void
tnlNeumannBoundaryConditions< tnlGrid< 2, MeshReal, Device, MeshIndex >, Vector, Real, Index >::
setBoundaryConditions( const RealType& time,
                       const MeshType& mesh,
                       const IndexType index,
                       const CoordinatesType& coordinates,
                       DofVectorType& u,
                       DofVectorType& fu ) const
{
   fu[ index ] = 0;
   if( coordinates.x() == 0 )
   {
      u[ index ] = u[ mesh.template getCellNextToCell< 1, 0 >( index ) ] - mesh.getHx() * this->vector[ index ];
      return;
   }
   if( coordinates.x() == mesh.getDimensions().x() - 1 )
   {
      u[ index ] = u[ mesh.template getCellNextToCell< -1, 0 >( index ) ] + mesh.getHx() * this->vector[ index ];
      return;
   }
   if( coordinates.y() == 0 )
   {
      u[ index ] = u[ mesh.template getCellNextToCell< 0, 1 >( index ) ] - mesh.getHy() * this->vector[ index ];
      return;
   }
   if( coordinates.y() == mesh.getDimensions().y() - 1 )
   {
      u[ index ] = u[ mesh.template getCellNextToCell< 0, -1 >( index ) ] + mesh.getHy() * this->vector[ index ];
      return;
   }
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Vector,
          typename Real,
          typename Index >
__cuda_callable__
Index
tnlNeumannBoundaryConditions< tnlGrid< 2, MeshReal, Device, MeshIndex >, Vector, Real, Index >::
getLinearSystemRowLength( const MeshType& mesh,
                          const IndexType& index,
                          const CoordinatesType& coordinates ) const
{
   return 2;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Vector,
          typename Real,
          typename Index >
   template< typename MatrixRow >
__cuda_callable__
void
tnlNeumannBoundaryConditions< tnlGrid< 2, MeshReal, Device, MeshIndex >, Vector, Real, Index >::
updateLinearSystem( const RealType& time,
                    const MeshType& mesh,
                    const IndexType& index,
                    const CoordinatesType& coordinates,
                    DofVectorType& u,
                    DofVectorType& b,
                    MatrixRow& matrixRow ) const
{
   if( coordinates.x() == 0 )
   {
      matrixRow.setElement( 0, index,                            1.0 );
      matrixRow.setElement( 1, mesh.template getCellNextToCell< 1, 0 >( index ), -1.0 );
      b[ index ] = - mesh.getHx() * this->vector[ index ];
   }
   if( coordinates.x() == mesh.getDimensions().x() - 1 )
   {
      matrixRow.setElement( 0, mesh.template getCellNextToCell< -1, 0 >( index ), -1.0 );
      matrixRow.setElement( 1, index,                              1.0 );
      b[ index ] = mesh.getHx() * this->vector[ index ];
   }
   if( coordinates.y() == 0 )
   {
      matrixRow.setElement( 0, index,                            1.0 );
      matrixRow.setElement( 1, mesh.template getCellNextToCell< 0, 1 >( index ), -1.0 );
      b[ index ] = - mesh.getHy() * this->vector[ index ];
   }
   if( coordinates.y() == mesh.getDimensions().y() - 1 )
   {
      matrixRow.setElement( 0, mesh.template getCellNextToCell< 0, -1 >( index ), -1.0 );
      matrixRow.setElement( 1, index,                              1.0 );
      b[ index ] = mesh.getHy() * this->vector[ index ];
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
__cuda_callable__
void
tnlNeumannBoundaryConditions< tnlGrid< 3, MeshReal, Device, MeshIndex >, Vector, Real, Index >::
setBoundaryConditions( const RealType& time,
                       const MeshType& mesh,
                       const IndexType index,
                       const CoordinatesType& coordinates,
                       DofVectorType& u,
                       DofVectorType& fu ) const
{
   fu[ index ] = 0;
   if( coordinates.x() == 0 )
   {
      u[ index ] = u[ mesh.template getCellNextToCell< 1, 0, 0 >( index ) ] - mesh.getHx() * this->vector[ index ];
      return;
   }
   if( coordinates.x() == mesh.getDimensions().x() - 1 )
   {
      u[ index ] = u[ mesh.template getCellNextToCell< -1, 0, 0 >( index ) ] + mesh.getHx() * this->vector[ index ];
      return;
   }
   if( coordinates.y() == 0 )
   {
      u[ index ] = u[ mesh.template getCellNextToCell< 0, 1, 0 >( index ) ] - mesh.getHy() * this->vector[ index ];
      return;
   }
   if( coordinates.y() == mesh.getDimensions().y() - 1 )
   {
      u[ index ] = u[ mesh.template getCellNextToCell< 0, -1, 0 >( index ) ] + mesh.getHy() * this->vector[ index ];
      return;
   }
   if( coordinates.z() == 0 )
   {
      u[ index ] = u[ mesh.template getCellNextToCell< 0, 0, 1 >( index ) ] - mesh.getHz() * this->vector[ index ];
      return;
   }
   if( coordinates.z() == mesh.getDimensions().z() - 1 )
   {
      u[ index ] = u[ mesh.template getCellNextToCell< 0, 0, -1 >( index ) ] + mesh.getHz() * this->vector[ index ];
      return;
   }
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Vector,
          typename Real,
          typename Index >
__cuda_callable__
Index
tnlNeumannBoundaryConditions< tnlGrid< 3, MeshReal, Device, MeshIndex >, Vector, Real, Index >::
getLinearSystemRowLength( const MeshType& mesh,
                          const IndexType& index,
                          const CoordinatesType& coordinates ) const
{
   return 2;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Vector,
          typename Real,
          typename Index >
   template< typename MatrixRow >
__cuda_callable__
void
tnlNeumannBoundaryConditions< tnlGrid< 3, MeshReal, Device, MeshIndex >, Vector, Real, Index >::
updateLinearSystem( const RealType& time,
                    const MeshType& mesh,
                    const IndexType& index,
                    const CoordinatesType& coordinates,
                    DofVectorType& u,
                    DofVectorType& b,
                    MatrixRow& matrixRow ) const
{
   if( coordinates.x() == 0 )
   {
      matrixRow.setElement( 0, index,                            1.0 );
      matrixRow.setElement( 1, mesh.template getCellNextToCell< 1, 0, 0 >( index ), -1.0 );
      b[ index ] = - mesh.getHx() * this->vector[ index ];
   }
   if( coordinates.x() == mesh.getDimensions().x() - 1 )
   {
      matrixRow.setElement( 0, mesh.template getCellNextToCell< -1, 0, 0 >( index ), -1.0 );
      matrixRow.setElement( 1, index,                              1.0 );
      b[ index ] = mesh.getHx() * this->vector[ index ];
   }
   if( coordinates.y() == 0 )
   {
      matrixRow.setElement( 0, index,                            1.0 );
      matrixRow.setElement( 1, mesh.template getCellNextToCell< 0, 1, 0 >( index ), -1.0 );
      b[ index ] = - mesh.getHy() * this->vector[ index ];
   }
   if( coordinates.y() == mesh.getDimensions().y() - 1 )
   {
      matrixRow.setElement( 0, mesh.template getCellNextToCell< 0, -1, 0 >( index ), -1.0 );
      matrixRow.setElement( 1, index,                              1.0 );
      b[ index ] = mesh.getHy() * this->vector[ index ];
   }
   if( coordinates.z() == 0 )
   {
      matrixRow.setElement( 0, index,                            1.0 );
      matrixRow.setElement( 1, mesh.template getCellNextToCell< 0, 0, 1 >( index ), -1.0 );
      b[ index ] = - mesh.getHz() * this->vector[ index ];
   }
   if( coordinates.z() == mesh.getDimensions().z() - 1 )
   {
      matrixRow.setElement( 0, mesh.template getCellNextToCell< 0, 0, -1 >( index ), -1.0 );
      matrixRow.setElement( 1, index,                              1.0 );
      b[ index ] = mesh.getHz() * this->vector[ index ];
   }
}

#endif	/* TNLNEUMANNBOUNDARYCONDITIONS_IMPL_H */

