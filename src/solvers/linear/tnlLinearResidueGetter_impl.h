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

template< typename Matrix, typename Vector >
typename tnlLinearResidueGetter< Matrix, Vector > :: RealType
   tnlLinearResidueGetter< Matrix, Vector > :: getResidue( const Matrix& matrix,
                                                           const Vector& x,
                                                           const Vector& b,
                                                           RealType bNorm )
{
   const IndexType size = matrix. getRows();
   RealType res( 0.0 );
   if( bNorm == 0.0 )
      bNorm = b. lpNorm( 2.0 );
   for( IndexType i = 0; i < size; i ++ )
   {
      RealType err = fabs( matrix. rowVectorProduct( i, x ) - b[ i ] );
      res += err * err;
   }
   if (bNorm!=0.0)  return sqrt( res ) / bNorm;
   return sqrt(res);
}

#endif /* TNLLINEARRESIDUEGETTER_IMPL_H_ */
