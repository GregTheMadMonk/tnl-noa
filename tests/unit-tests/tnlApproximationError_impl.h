/***************************************************************************
                          tnlApproximationError_impl.h  -  description
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

#ifndef TNLAPPROXIMATIONERROR_IMPL_H_
#define TNLAPPROXIMATIONERROR_IMPL_H_

#include <mesh/tnlTraversal.h>
#include <core/vectors/tnlVector.h>

template< typename Mesh,
          typename ExactOperator,
          typename ApproximateOperator,
          typename Function >
void
tnlApproximationError< Mesh, ExactOperator, ApproximateOperator, Function >::
getErrc( const Mesh& mesh,
         const ExactOperator& exactOperator,
         const ApproximateOperator& approximateOperator,
         const Function& function,
         RealType& l1Err,
         RealType& l2Err,
         RealType& maxErr )
{
   tnlVector< RealType, DeviceType, IndexType > exactOperator, approximateOperator;
   const IndexType entities = mesh.getNumberOfCells();

   if( ! exactOperator.setSize( entities ) ||
       ! approximateOperator.setSize( entities )  )
      return false;

   if( DeviceType::DeviceType == ( int ) tnlHostDevice )
   {
      for( IndexType i = 0; i < coarserEntities; i++ )
      {
         if( ! mesh.isBoundaryCell( i ) )
         {
            VertexType v = mesh.getCellCenter( i );
            coarserExactOperator[ i ] = exactOperator.getValue( function, v );
            coarserApproximateOperator[ i ] = approximateOperator.getValue( mesh, function, i );
         }
         else exactOperator[ i ] = approximateOperator[ i ];
      }
   }
   if( DeviceType::DeviceType == ( int ) tnlCudaDevice )
   {
      // TODO
   }
   L1Err = mesh.getDifferenceLpNorm( exactOperator, approximateOperator, ( RealType ) 1.0 );
   L2Errr = mesh.getDifferenceLpNorm( exactOperator, approximateOperator, ( RealType ) 2.0 );
   MaxErrr = mesh.getDifferenceAbsMax( exactOperator, approximateOperator );
}

#endif /* TNLAPPROXIMATIONERROR_IMPL_H_ */
