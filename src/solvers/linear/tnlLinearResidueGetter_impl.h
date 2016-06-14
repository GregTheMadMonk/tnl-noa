/***************************************************************************
                          tnlLinearResidueGetter_impl.h  -  description
                             -------------------
    begin                : Nov 25, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
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

#ifndef TNLLINEARRESIDUEGETTER_IMPL_H_
#define TNLLINEARRESIDUEGETTER_IMPL_H_

template< typename MatrixPointer, typename VectorPointer >
typename tnlLinearResidueGetter< MatrixPointer, VectorPointer >::RealType
tnlLinearResidueGetter< MatrixPointer, VectorPointer >::
getResidue( const MatrixPointer& matrix,
            const VectorPointer& xPtr,
            const VectorPointer& bPtr,
            RealType bNorm )
{
   typedef typename VectorPointer::ObjectType VectorType;
   const VectorType& x = *xPtr;
   const VectorType& b = *bPtr;
   const IndexType size = matrix->getRows();   
   RealType res( 0.0 );
   if( bNorm == 0.0 )
      bNorm = b.lpNorm( 2.0 );
   for( IndexType i = 0; i < size; i ++ )
   {
      RealType err = fabs( matrix->rowVectorProduct( i, x ) - b[ i ] );
      res += err * err;
   }
   return sqrt( res ) / bNorm;
}

#endif /* TNLLINEARRESIDUEGETTER_IMPL_H_ */
