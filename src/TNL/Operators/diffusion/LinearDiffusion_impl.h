/***************************************************************************
                          LinearDiffusion_impl.h  -  description
                             -------------------
    begin                : Aug 8, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Operators/diffusion/LinearDiffusion.h>
#include <TNL/Meshes/Grid.h>

namespace TNL {
namespace Operators {   

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
String
LinearDiffusion< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >::
getType()
{
   return String( "LinearDiffusion< " ) +
          MeshType::getType() + ", " +
         TNL::getType< Real >() + ", " +
         TNL::getType< Index >() + " >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename PreimageFunction,
          typename MeshEntity >
__cuda_callable__
inline
Real
LinearDiffusion< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >::
operator()( const PreimageFunction& u,
            const MeshEntity& entity,
            const Real& time ) const
{
   static_assert( MeshEntity::entityDimensions == 1, "Wrong mesh entity dimensions." );
   static_assert( PreimageFunction::getEntitiesDimensions() == 1, "Wrong preimage function" );
   const typename MeshEntity::template NeighbourEntities< 1 >& neighbourEntities = entity.getNeighbourEntities();
   const RealType& hxSquareInverse = entity.getMesh().template getSpaceStepsProducts< - 2 >();
   return ( u[ neighbourEntities.template getEntityIndex< -1 >() ]
            - 2.0 * u[ entity.getIndex() ]
            + u[ neighbourEntities.template getEntityIndex< 1 >() ] ) * hxSquareInverse;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
   template< typename MeshEntity >
__cuda_callable__
inline
Index
LinearDiffusion< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >::
getLinearSystemRowLength( const MeshType& mesh,
                          const IndexType& index,
                          const MeshEntity& entity ) const
{
   return 3;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
   template< typename PreimageFunction,
             typename MeshEntity,
             typename Matrix,
             typename Vector >
__cuda_callable__
inline
void
LinearDiffusion< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >::
setMatrixElements( const PreimageFunction& u,
                   const MeshEntity& entity,
                   const RealType& time,
                   const RealType& tau,
                   Matrix& matrix,
                   Vector& b ) const
{
   static_assert( MeshEntity::entityDimensions == 1, "Wrong mesh entity dimensions." );
   static_assert( PreimageFunction::getEntitiesDimensions() == 1, "Wrong preimage function" );
   const typename MeshEntity::template NeighbourEntities< 1 >& neighbourEntities = entity.getNeighbourEntities();
   const IndexType& index = entity.getIndex();
   typename Matrix::MatrixRow matrixRow = matrix.getRow( index );
   const RealType lambdaX = tau * entity.getMesh().template getSpaceStepsProducts< -2 >();
   matrixRow.setElement( 0, neighbourEntities.template getEntityIndex< -1 >(),      - lambdaX );
   matrixRow.setElement( 1, index,                                              2.0 * lambdaX );
   matrixRow.setElement( 2, neighbourEntities.template getEntityIndex< 1 >(),       - lambdaX );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
String
LinearDiffusion< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
getType()
{
   return String( "LinearDiffusion< " ) +
          MeshType::getType() + ", " +
         TNL::getType< Real >() + ", " +
         TNL::getType< Index >() + " >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
   template< typename EntityType >
__cuda_callable__
inline
Index
LinearDiffusion< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
getLinearSystemRowLength( const MeshType& mesh,
                          const IndexType& index,
                          const EntityType& entity ) const
{
   return 5;
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename PreimageFunction,
          typename EntityType >
__cuda_callable__
inline
Real
LinearDiffusion< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
operator()( const PreimageFunction& u,
            const EntityType& entity,
            const Real& time ) const
{  
   static_assert( EntityType::entityDimensions == 2, "Wrong mesh entity dimensions." );
   static_assert( PreimageFunction::getEntitiesDimensions() == 2, "Wrong preimage function" );
  // std::cerr << "Lin Diff op:" <<__LINE__ << std::endl<< std::flush;
   const typename EntityType::template NeighbourEntities< 2 >& neighbourEntities = entity.getNeighbourEntities();
  // std::cerr << "Lin Diff op:" <<__LINE__ << std::endl<< std::flush;
   const RealType& hxSquareInverse = entity.getMesh().template getSpaceStepsProducts< -2, 0 >();
 //  std::cerr << "Lin Diff op:" <<__LINE__ << std::endl<< std::flush;
   const RealType& hySquareInverse = entity.getMesh().template getSpaceStepsProducts< 0, -2 >();
 //  std::cerr << "Lin Diff op:" <<__LINE__ << std::endl<< std::flush;
   
   //int stop;
   //std::cin >> stop;
   
 /*  std::cerr << "Lin Diff indexes:" <<__LINE__ << "\t"
           << neighbourEntities.template getEntityIndex< -1,  0 >()<<"\t"
           << neighbourEntities.template getEntityIndex< 1,  0 >()<<"\t"
           <<neighbourEntities.template getEntityIndex< 0,  -1 >()<<"\t"
           <<neighbourEntities.template getEntityIndex< 0,  1 >()<<"\t"
           <<std::endl<< std::flush;*/
   
   //std::cin >> stop;
   return ( u[ neighbourEntities.template getEntityIndex< -1,  0 >() ]
          + u[ neighbourEntities.template getEntityIndex<  1,  0 >() ] ) * hxSquareInverse +
          ( u[ neighbourEntities.template getEntityIndex<  0, -1 >() ]
          + u[ neighbourEntities.template getEntityIndex<  0,  1 >() ] ) * hySquareInverse
          - 2.0 * u[ entity.getIndex() ] * ( hxSquareInverse + hySquareInverse );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
   template< typename PreimageFunction,
             typename MeshEntity,
             typename Matrix,
             typename Vector >
__cuda_callable__
inline
void
LinearDiffusion< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
setMatrixElements( const PreimageFunction& u,
                   const MeshEntity& entity,
                   const RealType& time,
                   const RealType& tau,
                   Matrix& matrix,
                   Vector& b ) const
{
   static_assert( MeshEntity::entityDimensions == 2, "Wrong mesh entity dimensions." );
   static_assert( PreimageFunction::getEntitiesDimensions() == 2, "Wrong preimage function" );
   const IndexType& index = entity.getIndex();
   typename Matrix::MatrixRow matrixRow = matrix.getRow( index );
   const RealType lambdaX = tau * entity.getMesh().template getSpaceStepsProducts< -2, 0 >();
   const RealType lambdaY = tau * entity.getMesh().template getSpaceStepsProducts< 0, -2 >();
   const typename MeshEntity::template NeighbourEntities< 2 >& neighbourEntities = entity.getNeighbourEntities();
   matrixRow.setElement( 0, neighbourEntities.template getEntityIndex< 0, -1 >(), -lambdaY );
   matrixRow.setElement( 1, neighbourEntities.template getEntityIndex< -1, 0 >(), -lambdaX );
   matrixRow.setElement( 2, index,                                                        2.0 * ( lambdaX + lambdaY ) );
   matrixRow.setElement( 3, neighbourEntities.template getEntityIndex< 1, 0 >(),   -lambdaX );
   matrixRow.setElement( 4, neighbourEntities.template getEntityIndex< 0, 1 >(),   -lambdaY );
}


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
String
LinearDiffusion< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >::
getType()
{
   return String( "LinearDiffusion< " ) +
          MeshType::getType() + ", " +
         TNL::getType< Real >() + ", " +
         TNL::getType< Index >() + " >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename PreimageFunction,
          typename EntityType >
__cuda_callable__
inline
Real
LinearDiffusion< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >::
operator()( const PreimageFunction& u,
            const EntityType& entity,
            const Real& time ) const
{
   static_assert( EntityType::entityDimensions == 3, "Wrong mesh entity dimensions." );
   static_assert( PreimageFunction::getEntitiesDimensions() == 3, "Wrong preimage function" );
   const typename EntityType::template NeighbourEntities< 3 >& neighbourEntities = entity.getNeighbourEntities();
   const RealType& hxSquareInverse = entity.getMesh().template getSpaceStepsProducts< -2,  0,  0 >();
   const RealType& hySquareInverse = entity.getMesh().template getSpaceStepsProducts<  0, -2,  0 >();
   const RealType& hzSquareInverse = entity.getMesh().template getSpaceStepsProducts<  0,  0, -2 >();
   return (   u[ neighbourEntities.template getEntityIndex< -1,  0,  0 >() ]
            + u[ neighbourEntities.template getEntityIndex<  1,  0,  0 >() ] ) * hxSquareInverse +
          (   u[ neighbourEntities.template getEntityIndex<  0, -1,  0 >() ]
            + u[ neighbourEntities.template getEntityIndex<  0,  1,  0 >() ] ) * hySquareInverse +
          (   u[ neighbourEntities.template getEntityIndex<  0,  0, -1 >() ]
            + u[ neighbourEntities.template getEntityIndex<  0,  0,  1 >() ] ) * hzSquareInverse
         - 2.0 * u[ entity.getIndex() ] * ( hxSquareInverse + hySquareInverse + hzSquareInverse );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
   template< typename EntityType >
__cuda_callable__
inline
Index
LinearDiffusion< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >::
getLinearSystemRowLength( const MeshType& mesh,
                          const IndexType& index,
                          const EntityType& entity ) const
{
   return 7;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
   template< typename PreimageFunction,
             typename MeshEntity,
             typename Matrix,
             typename Vector >
__cuda_callable__
inline
void
LinearDiffusion< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >::
setMatrixElements( const PreimageFunction& u,
                   const MeshEntity& entity,
                   const RealType& time,
                   const RealType& tau,
                   Matrix& matrix,
                   Vector& b ) const
{
   static_assert( MeshEntity::entityDimensions == 3, "Wrong mesh entity dimensions." );
   static_assert( PreimageFunction::getEntitiesDimensions() == 3, "Wrong preimage function" );
   const typename MeshEntity::template NeighbourEntities< 3 >& neighbourEntities = entity.getNeighbourEntities();
   const IndexType& index = entity.getIndex();
   typename Matrix::MatrixRow matrixRow = matrix.getRow( index );
   const RealType lambdaX = tau * entity.getMesh().template getSpaceStepsProducts< -2, 0, 0 >();
   const RealType lambdaY = tau * entity.getMesh().template getSpaceStepsProducts< 0, -2, 0 >();
   const RealType lambdaZ = tau * entity.getMesh().template getSpaceStepsProducts< 0, 0, -2 >();
   matrixRow.setElement( 0, neighbourEntities.template getEntityIndex< 0, 0, -1 >(), -lambdaZ );
   matrixRow.setElement( 1, neighbourEntities.template getEntityIndex< 0, -1, 0 >(), -lambdaY );
   matrixRow.setElement( 2, neighbourEntities.template getEntityIndex< -1, 0, 0 >(), -lambdaX );
   matrixRow.setElement( 3, index,                             2.0 * ( lambdaX + lambdaY + lambdaZ ) );
   matrixRow.setElement( 4, neighbourEntities.template getEntityIndex< 1, 0, 0 >(),   -lambdaX );
   matrixRow.setElement( 5, neighbourEntities.template getEntityIndex< 0, 1, 0 >(),   -lambdaY );
   matrixRow.setElement( 6, neighbourEntities.template getEntityIndex< 0, 0, 1 >(),   -lambdaZ );
}

} // namespace Operators
} // namespace TNL
