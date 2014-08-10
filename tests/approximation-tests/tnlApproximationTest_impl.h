/***************************************************************************
                          tnlApproximationTest_impl.h  -  description
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

#ifndef TNLAPPROXIMATIONTEST_IMPL_H_
#define TNLAPPROXIMATIONTEST_IMPL_H_

#include <mesh/tnlTraversal.h>
#include <core/vectors/tnlVector.h>

template< typename Mesh,
          typename ExactOperator,
          typename ApproximateOperator,
          typename Function >
void
tnlApproximationTest< Mesh, ExactOperator, ApproximateOperator, Function >::
getEoc( const Mesh& coarserMesh,
        const Mesh& finerMesh,
        const RealType& refinement,
        const ExactOperator& exactOperator,
        const ApproximateOperator& approximateOperator,
        const Function& function,
        RealType& l1Eoc,
        RealType& l2Eoc,
        RealType& maxEoc )
{
   tnlVector< RealType, DeviceType, IndexType > coarserExactOperator, coarserApproximateOperator,
                                                finerExactOperator, finerApproximateOperator;
   const IndexType coarserEntities = coarserMesh.getNumberOfCells();
   const IndexType finerEntities = finerMesh.getNumberOfCells();

   if( ! coarserExactOperator.setSize( coarserEntities ) ||
       ! coarserApproximateOperator.setSize( coarserEntities ) ||
       ! finerExactOperator.setSize( finerEntities ) ||
       ! finerApproximateOperator.setSize( finerEntities ) )
      return false;

   if( DeviceType::DeviceType == ( int ) tnlHostDevice )
   {
      for( IndexType i = 0; i < coarserEntities; i++ )
      {
         if( ! coarserMesh.isBoundaryCell( i ) )
         {
            VertexType v = coarserMesh.getCellCenter( i );
            coarserExactOperator[ i ] = exactOperator.getValue( function, v );
            coarserApproximateOperator[ i ] = approximateOperator.getValue( coarserMesh, function, i );
         }
         else coarserExactOperator[ i ] = coarserApproximateOperator[ i ];
      }
      for( IndexType i = 0; i < finerEntities; i++ )
      {
         if( ! coarserMesh.isBoundaryCell( i ) )
         {
            VertexType v = finerMesh.getCellCenter( i );
            finerExactOperator[ i ] = exactOperator.getValue( function, v );
            finerApproximateOperator[ i ] = approximateOperator.getValue( finerMesh, function, i );
         }
         else finerExactOperator[ i ] = finerApproximateOperator[ i ];
      }
   }
   if( DeviceType::DeviceType == ( int ) tnlCudaDevice )
   {
      // TODO
   }
   const RealType& coarserL1Error = coarserMesh.getDifferenceLpNorm( coarserExactOperator, coarserApproximateOperator, ( RealType ) 1.0 );
   const RealType& coarserL2Error = coarserMesh.getDifferenceLpNorm( coarserExactOperator, coarserApproximateOperator, ( RealType ) 2.0 );
   const RealType& coarserMaxError = coarserMesh.getDifferenceAbsMax( coarserExactOperator, coarserApproximateOperator, ( RealType ) );

   const RealType& finerL1Error = finerMesh.getDifferenceLpNorm( finerExactOperator, finerApproximateOperator, ( RealType ) 1.0 );
   const RealType& finerL2Error = finerMesh.getDifferenceLpNorm( finerExactOperator, finerApproximateOperator, ( RealType ) 2.0 );
   const RealType& finerMaxError = finerMesh.getDifferenceAbsMax( finerExactOperator, finerApproximateOperator, ( RealType ) );

   l1Eoc = ln( coarserL1Error / finerL1Error ) / ln( refinement );
   l2Eoc = ln( coarserL2Error / finerL2Error ) / ln( refinement );
   maxEoc = ln( coarserMaxError / finerMaxError ) / ln( refinement );
}

#endif /* TNLAPPROXIMATIONTEST_IMPL_H_ */
