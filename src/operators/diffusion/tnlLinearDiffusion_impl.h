/***************************************************************************
                          tnlLinearDiffusion_impl.h  -  description
                             -------------------
    begin                : Aug 8, 2014
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

#ifndef TNLLINEARDIFFUSION_IMP_H
#define	TNLLINEARDIFFUSION_IMP_H

#include <operators/diffusion/tnlLinearDiffusion.h>
#include <mesh/tnlGrid.h>

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
tnlString
tnlLinearDiffusion< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index >::
getType()
{
   return tnlString( "tnlLinearDiffusion< " ) +
          MeshType::getType() + ", " +
          ::getType< Real >() + ", " +
          ::getType< Index >() + " >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename Vector >
__cuda_callable__
inline
Real
tnlLinearDiffusion< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index >::
getValue( const MeshType& mesh,
          const IndexType cellIndex,
          const CellType& cell,
          const Vector& u,
          const Real& time ) const
{
   return ( u[ mesh.template getCellNextToCell< -1 >( cellIndex ) ]
            - 2.0 * u[ cellIndex ]
            + u[ mesh.template getCellNextToCell< 1 >( cellIndex ) ] ) * mesh.getHxSquareInverse();
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
__cuda_callable__
inline
Index
tnlLinearDiffusion< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index >::
getLinearSystemRowLength( const MeshType& mesh,
                          const IndexType& index,
                          const CellType& cell ) const
{
   return 3;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
   template< typename Vector, typename Matrix >
__cuda_callable__
inline
void
tnlLinearDiffusion< tnlGrid< 1, MeshReal, Device, MeshIndex >, Real, Index >::
updateLinearSystem( const RealType& time,
                    const RealType& tau,
                    const MeshType& mesh,
                    const IndexType& index,
                    const CellType& cell,
                    Vector& u,
                    Vector& b,
                    Matrix& matrix ) const
{
   typename Matrix::MatrixRow matrixRow = matrix.getRow( index );
   const RealType lambdaX = tau * mesh.getHxSquareInverse();
   //printf( "tau = %f lambda = %f dx_sqr = %f dx = %f, \n", tau, lambdaX, mesh.getHxSquareInverse(), mesh.getHx() );
   matrixRow.setElement( 0, mesh.template getCellNextToCell< -1 >( index ),     - lambdaX );
   matrixRow.setElement( 1, index,                             2.0 * lambdaX );
   matrixRow.setElement( 2, mesh.template getCellNextToCell< 1 >( index ),       - lambdaX );
   //printf( "Linear diffusion index %d columns %d %d %d \n", index, columns[ 0 ], columns[ 1 ], columns[ 2 ] );
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
tnlString
tnlLinearDiffusion< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
getType()
{
   return tnlString( "tnlLinearDiffusion< " ) +
          MeshType::getType() + ", " +
          ::getType< Real >() + ", " +
          ::getType< Index >() + " >";
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
tnlLinearDiffusion< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
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
template< typename Vector,
          typename EntityType >
__cuda_callable__
inline
Real
tnlLinearDiffusion< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
getValue( const MeshType& mesh,
          const IndexType cellIndex,
          const EntityType& entity,
          const Vector& u,
          const Real& time ) const
{
   auto neighbourEntities = entity.getNeighbourEntities();
   return ( u[ neighbourEntities.template getEntityIndex< -1, 0 >() ]
            - 2.0 * u[ cellIndex ]
            + u[ neighbourEntities.template getEntityIndex< 1, 0 >() ] ) * mesh.getHxSquareInverse() +
           ( u[ neighbourEntities.template getEntityIndex< 0, -1 >() ]
             - 2.0 * u[ cellIndex ]
             + u[ neighbourEntities.template getEntityIndex< 0, 1 >() ] ) * mesh.getHySquareInverse();
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
   template< typename Vector,
             typename Matrix,
             typename EntityType >
__cuda_callable__
inline
void
tnlLinearDiffusion< tnlGrid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
updateLinearSystem( const RealType& time,
                    const RealType& tau,
                    const MeshType& mesh,
                    const IndexType& index,
                    const EntityType& entity,
                    Vector& u,
                    Vector& b,
                    Matrix& matrix ) const
{
   typename Matrix::MatrixRow matrixRow = matrix.getRow( index );
   const RealType lambdaX = tau * mesh.getHxSquareInverse();
   const RealType lambdaY = tau * mesh.getHySquareInverse();
   auto neighbourEntities = entity.getNeighbourEntities();
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
tnlString
tnlLinearDiffusion< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index >::
getType()
{
   return tnlString( "tnlLinearDiffusion< " ) +
          MeshType::getType() + ", " +
          ::getType< Real >() + ", " +
          ::getType< Index >() + " >";
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename Vector,
          typename EntityType >
__cuda_callable__
inline
Real
tnlLinearDiffusion< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index >::
getValue( const MeshType& mesh,
          const IndexType cellIndex,
          const EntityType& entity,
          const Vector& u,
          const Real& time ) const
{
   return ( u[ mesh.template getCellNextToCell< -1, 0, 0 >( cellIndex ) ]
            - 2.0 * u[ cellIndex ]
            + u[ mesh.template getCellNextToCell< 1, 0, 0 >( cellIndex ) ] ) * mesh.getHxSquareInverse() +
          ( u[ mesh.template getCellNextToCell< 0, -1, 0 >( cellIndex ) ]
            - 2.0 * u[ cellIndex ]
            + u[ mesh.template getCellNextToCell< 0, 1, 0 >( cellIndex ) ] ) * mesh.getHySquareInverse() +
          ( u[ mesh.template getCellNextToCell< 0, 0, -1 >( cellIndex ) ]
            - 2.0 * u[ cellIndex ]
            + u[ mesh.template getCellNextToCell< 0, 0, 1 >( cellIndex ) ] ) * mesh.getHzSquareInverse();
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
tnlLinearDiffusion< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index >::
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
   template< typename Vector,
             typename Matrix,
             typename EntityType >
__cuda_callable__
inline
void
tnlLinearDiffusion< tnlGrid< 3, MeshReal, Device, MeshIndex >, Real, Index >::
updateLinearSystem( const RealType& time,
                    const RealType& tau,
                    const MeshType& mesh,
                    const IndexType& index,
                    const EntityType& entity,
                    Vector& u,
                    Vector& b,
                    Matrix& matrix ) const
{
   typename Matrix::MatrixRow matrixRow = matrix.getRow( index );
   const RealType lambdaX = tau * mesh.getHxSquareInverse();
   const RealType lambdaY = tau * mesh.getHySquareInverse();
   const RealType lambdaZ = tau * mesh.getHzSquareInverse();
   matrixRow.setElement( 0, mesh.template getCellNextToCell< 0, 0, -1 >( index ), -lambdaZ );
   matrixRow.setElement( 1, mesh.template getCellNextToCell< 0, -1, 0 >( index ), -lambdaY );
   matrixRow.setElement( 2, mesh.template getCellNextToCell< -1, 0, 0 >( index ), -lambdaX );
   matrixRow.setElement( 3, index,                             2.0 * ( lambdaX + lambdaY + lambdaZ ) );
   matrixRow.setElement( 4, mesh.template getCellNextToCell< 1, 0, 0 >( index ),   -lambdaX );
   matrixRow.setElement( 5, mesh.template getCellNextToCell< 0, 1, 0 >( index ),   -lambdaY );
   matrixRow.setElement( 6, mesh.template getCellNextToCell< 0, 0, 1 >( index ),   -lambdaZ );
}

#endif	/* TNLLINEARDIFFUSION_IMP_H */
