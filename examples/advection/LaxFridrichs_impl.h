#ifndef LaxFridrichs_IMPL_H
#define LaxFridrichs_IMPL_H

namespace TNL {

/****
 * 1D problem
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
String
LaxFridrichs< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >::
getType()
{
   return String( "LaxFridrichs< " ) +
          MeshType::getType() + ", " +
          TNL::getType< Real >() + ", " +
          TNL::getType< Index >() + " >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshFunction, typename MeshEntity >
__cuda_callable__
Real
LaxFridrichs< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >::
operator()( const MeshFunction& u,
            const MeshEntity& entity,
            const Real& time ) const
{
   /****
    * Implement your explicit form of the differential operator here.
    * The following example is the Laplace operator approximated 
    * by the Finite difference method.
    */
    static_assert( MeshEntity::entityDimensions == 1, "Wrong mesh entity dimensions." ); 
    static_assert( MeshFunction::getEntitiesDimensions() == 1, "Wrong preimage function" ); 
    const typename MeshEntity::template NeighbourEntities< 1 >& neighbourEntities = entity.getNeighbourEntities(); 

   const RealType& hxInverse = entity.getMesh().template getSpaceStepsProducts< -1 >(); 
   const IndexType& center = entity.getIndex(); 
   const IndexType& east = neighbourEntities.template getEntityIndex< 1 >(); 
   const IndexType& west = neighbourEntities.template getEntityIndex< -1 >(); 
   return   (0.5 / this->tau ) * this->artificalViscosity *
	    ( u[ west ]- 2.0 * u[ center ] + u[ east ] )
            - (this->advectionSpeedX [ east ] * u[ east ] - this->advectionSpeedX [ west ] * u[west] ) * hxInverse * 0.5;

}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshEntity >
__cuda_callable__
Index
LaxFridrichs< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >::
getLinearSystemRowLength( const MeshType& mesh,
                          const IndexType& index,
                          const MeshEntity& entity ) const
{
   /****
    * Return a number of non-zero elements in a line (associated with given grid element) of
    * the linear system.
    * The following example is the Laplace operator approximated 
    * by the Finite difference method.
    */

   return 2*Dimensions + 1;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
   template< typename MeshEntity, typename Vector, typename MatrixRow >
__cuda_callable__
void
LaxFridrichs< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >::
updateLinearSystem( const RealType& time,
                    const RealType& tau,
                    const MeshType& mesh,
                    const IndexType& index,
                    const MeshEntity& entity,
                    const MeshFunctionType& u,
                    Vector& b,
                    MatrixRow& matrixRow ) const
{
   /****
    * Setup the non-zero elements of the linear system here.
    * The following example is the Laplace operator appriximated 
    * by the Finite difference method.
    */

    const typename MeshEntity::template NeighbourEntities< 1 >& neighbourEntities = entity.getNeighbourEntities(); 
   const RealType& lambdaX = tau * entity.getMesh().template getSpaceStepsProducts< -2 >(); 
   const IndexType& center = entity.getIndex(); 
   const IndexType& east = neighbourEntities.template getEntityIndex< 1 >(); 
   const IndexType& west = neighbourEntities.template getEntityIndex< -1 >(); 
   matrixRow.setElement( 0, west,   - lambdaX );
   matrixRow.setElement( 1, center, 2.0 * lambdaX );
   matrixRow.setElement( 2, east,   - lambdaX );
}

/****
 * 2D problem
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
String
LaxFridrichs< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
getType()
{
   return String( "LaxFridrichs< " ) +
          MeshType::getType() + ", " +
          TNL::getType< Real >() + ", " +
          TNL::getType< Index >() + " >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshFunction, typename MeshEntity >
__cuda_callable__
Real
LaxFridrichs< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
operator()( const MeshFunction& u,
            const MeshEntity& entity,
            const Real& time ) const
{
   /****
    * Implement your explicit form of the differential operator here.
    * The following example is the Laplace operator approximated 
    * by the Finite difference method.
    */
    static_assert( MeshEntity::entityDimensions == 2, "Wrong mesh entity dimensions." ); 
    static_assert( MeshFunction::getEntitiesDimensions() == 2, "Wrong preimage function" ); 
    const typename MeshEntity::template NeighbourEntities< 2 >& neighbourEntities = entity.getNeighbourEntities(); 

   const RealType& hxInverse = entity.getMesh().template getSpaceStepsProducts< -1, 0 >(); 
   const RealType& hyInverse = entity.getMesh().template getSpaceStepsProducts< 0, -1 >(); 
   const IndexType& center = entity.getIndex(); 
   const IndexType& east  = neighbourEntities.template getEntityIndex<  1,  0 >(); 
   const IndexType& west  = neighbourEntities.template getEntityIndex< -1,  0 >(); 
   const IndexType& north = neighbourEntities.template getEntityIndex<  0,  1 >(); 
   const IndexType& south = neighbourEntities.template getEntityIndex<  0, -1 >(); 
   return ( 0.25 / this->tau ) * this->artificalViscosity * 
          ( u[ west ] + u[ east ] + u[ south ] + u[ north ] - 4 * u[ center ] ) -
          (this->advectionSpeedX [ east ] * u[ east ] - this->advectionSpeedX [ west ] * u[ west] ) * hxInverse * 0.5 - 
          (this->advectionSpeedY [ north ] * u[ north ] - this->advectionSpeedY [ south ] * u[ south ] ) * hyInverse * 0.5;


}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshEntity >
__cuda_callable__
Index
LaxFridrichs< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
getLinearSystemRowLength( const MeshType& mesh,
                          const IndexType& index,
                          const MeshEntity& entity ) const
{
   /****
    * Return a number of non-zero elements in a line (associated with given grid element) of
    * the linear system.
    * The following example is the Laplace operator approximated 
    * by the Finite difference method.
    */

   return 2*Dimensions + 1;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
   template< typename MeshEntity, typename Vector, typename MatrixRow >
__cuda_callable__
void
LaxFridrichs< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
updateLinearSystem( const RealType& time,
                    const RealType& tau,
                    const MeshType& mesh,
                    const IndexType& index,
                    const MeshEntity& entity,
                    const MeshFunctionType& u,
                    Vector& b,
                    MatrixRow& matrixRow ) const
{
   /****
    * Setup the non-zero elements of the linear system here.
    * The following example is the Laplace operator appriximated 
    * by the Finite difference method.
    */

    const typename MeshEntity::template NeighbourEntities< 2 >& neighbourEntities = entity.getNeighbourEntities(); 
   const RealType& lambdaX = tau * entity.getMesh().template getSpaceStepsProducts< -2, 0 >(); 
   const RealType& lambdaY = tau * entity.getMesh().template getSpaceStepsProducts< 0, -2 >(); 
   const IndexType& center = entity.getIndex(); 
   const IndexType& east  = neighbourEntities.template getEntityIndex<  1,  0 >(); 
   const IndexType& west  = neighbourEntities.template getEntityIndex< -1,  0 >(); 
   const IndexType& north = neighbourEntities.template getEntityIndex<  0,  1 >(); 
   const IndexType& south = neighbourEntities.template getEntityIndex<  0, -1 >(); 
   matrixRow.setElement( 0, south,  -lambdaY );
   matrixRow.setElement( 1, west,   -lambdaX );
   matrixRow.setElement( 2, center, 2.0 * ( lambdaX + lambdaY ) );
   matrixRow.setElement( 3, east,   -lambdaX );
   matrixRow.setElement( 4, north,  -lambdaY );
}

/****
 * 3D problem
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
String
LaxFridrichs< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >::
getType()
{
   return String( "LaxFridrichs< " ) +
          MeshType::getType() + ", " +
         TNL::getType< Real >() + ", " +
         TNL::getType< Index >() + " >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshFunction, typename MeshEntity >
__cuda_callable__
Real
LaxFridrichs< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >::
operator()( const MeshFunction& u,
            const MeshEntity& entity,
            const Real& time ) const
{
   /****
    * Implement your explicit form of the differential operator here.
    * The following example is the Laplace operator approximated 
    * by the Finite difference method.
    */
    static_assert( MeshEntity::entityDimensions == 3, "Wrong mesh entity dimensions." ); 
    static_assert( MeshFunction::getEntitiesDimensions() == 3, "Wrong preimage function" ); 
    const typename MeshEntity::template NeighbourEntities< 3 >& neighbourEntities = entity.getNeighbourEntities(); 

   const RealType& hxSquareInverse = entity.getMesh().template getSpaceStepsProducts< -2,  0,  0 >(); 
   const RealType& hySquareInverse = entity.getMesh().template getSpaceStepsProducts<  0, -2,  0 >(); 
   const RealType& hzSquareInverse = entity.getMesh().template getSpaceStepsProducts<  0,  0, -2 >(); 
   const IndexType& center = entity.getIndex(); 
   const IndexType& east  = neighbourEntities.template getEntityIndex<  1,  0,  0 >(); 
   const IndexType& west  = neighbourEntities.template getEntityIndex< -1,  0,  0 >(); 
   const IndexType& north = neighbourEntities.template getEntityIndex<  0,  1,  0 >(); 
   const IndexType& south = neighbourEntities.template getEntityIndex<  0, -1,  0 >(); 
   const IndexType& up    = neighbourEntities.template getEntityIndex<  0,  0,  1 >(); 
   const IndexType& down  = neighbourEntities.template getEntityIndex<  0,  0, -1 >(); 
   return ( u[ west ] - 2.0 * u[ center ] + u[ east ]  ) * hxSquareInverse +
          ( u[ south ] - 2.0 * u[ center ] + u[ north ] ) * hySquareInverse +
          ( u[ up ] - 2.0 * u[ center ] + u[ down ] ) * hzSquareInverse;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshEntity >
__cuda_callable__
Index
LaxFridrichs< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >::
getLinearSystemRowLength( const MeshType& mesh,
                          const IndexType& index,
                          const MeshEntity& entity ) const
{
   /****
    * Return a number of non-zero elements in a line (associated with given grid element) of
    * the linear system.
    * The following example is the Laplace operator approximated 
    * by the Finite difference method.
    */

   //return 2*Dimensions + 1;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
   template< typename MeshEntity, typename Vector, typename MatrixRow >
__cuda_callable__
void
LaxFridrichs< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >::
updateLinearSystem( const RealType& time,
                    const RealType& tau,
                    const MeshType& mesh,
                    const IndexType& index,
                    const MeshEntity& entity,
                    const MeshFunctionType& u,
                    Vector& b,
                    MatrixRow& matrixRow ) const
{
   /****
    * Setup the non-zero elements of the linear system here.
    * The following example is the Laplace operator appriximated 
    * by the Finite difference method.
    */

   /* const typename MeshEntity::template NeighbourEntities< 3 >& neighbourEntities = entity.getNeighbourEntities(); 
   const RealType& lambdaX = tau * entity.getMesh().template getSpaceStepsProducts< -2,  0,  0 >(); 
   const RealType& lambdaY = tau * entity.getMesh().template getSpaceStepsProducts<  0, -2,  0 >(); 
   const RealType& lambdaZ = tau * entity.getMesh().template getSpaceStepsProducts<  0,  0, -2 >(); 
   const IndexType& center = entity.getIndex(); 
   const IndexType& east  = neighbourEntities.template getEntityIndex<  1,  0,  0 >(); 
   const IndexType& west  = neighbourEntities.template getEntityIndex< -1,  0,  0 >(); 
   const IndexType& north = neighbourEntities.template getEntityIndex<  0,  1,  0 >(); 
   const IndexType& south = neighbourEntities.template getEntityIndex<  0, -1,  0 >(); 
   const IndexType& up    = neighbourEntities.template getEntityIndex<  0,  0,  1 >(); 
   const IndexType& down  = neighbourEntities.template getEntityIndex<  0,  0, -1 >(); 
   matrixRow.setElement( 0, down,   -lambdaZ );
   matrixRow.setElement( 1, south,  -lambdaY );
   matrixRow.setElement( 2, west,   -lambdaX );
   matrixRow.setElement( 3, center, 2.0 * ( lambdaX + lambdaY + lambdaZ ) );
   matrixRow.setElement( 4, east,   -lambdaX );
   matrixRow.setElement( 5, north,  -lambdaY );
   matrixRow.setElement( 6, up,     -lambdaZ );*/
}

} // namespace TNL

#endif	/* LaxFridrichsIMPL_H */

