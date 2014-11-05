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
#include <functions/tnlFunctionDiscretizer.h>

template< typename Mesh,
          typename ExactOperator,
          typename ApproximateOperator,
          typename Function >
void
tnlApproximationError< Mesh, ExactOperator, ApproximateOperator, Function >::
getError( const Mesh& mesh,
          const ExactOperator& exactOperator,
          const ApproximateOperator& approximateOperator,
          const Function& function,
          RealType& l1Err,
          RealType& l2Err,
          RealType& maxErr )
{
   typedef tnlVector< RealType, DeviceType, IndexType > Vector;
   Vector functionData, exactData, approximateData;
   const IndexType entities = mesh.getNumberOfCells();

   if( ! functionData.setSize( entities ) ||
       ! exactData.setSize( entities ) ||
       ! approximateData.setSize( entities )  )
      return;

   tnlFunctionDiscretizer< Mesh, Function, Vector >::template discretize< 0, 0, 0 >( mesh, function, functionData );

   if( DeviceType::DeviceType == ( int ) tnlHostDevice )
   {
      for( IndexType i = 0; i < entities; i++ )
      {
         if( ! mesh.isBoundaryCell( i ) )
         {
            VertexType v = mesh.getCellCenter( i );
            CoordinatesType c = mesh.getCellCoordinates( i );
            exactData[ i ] = exactOperator.getValue( function, v );
            approximateData[ i ] = approximateOperator.getValue( mesh, i, c, functionData, 0.0 );
         }
         else exactData[ i ] = approximateData[ i ];
      }
   }
   if( DeviceType::DeviceType == ( int ) tnlCudaDevice )
   {
      // TODO
   }
   l1Err = mesh.getDifferenceLpNorm( exactData, approximateData, ( RealType ) 1.0 );
   l2Err = mesh.getDifferenceLpNorm( exactData, approximateData, ( RealType ) 2.0 );
   maxErr = mesh.getDifferenceAbsMax( exactData, approximateData );
}

#endif /* TNLAPPROXIMATIONERROR_IMPL_H_ */
