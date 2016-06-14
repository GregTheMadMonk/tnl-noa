/***************************************************************************
                          tnlLinearResidueGetter.h  -  description
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

#ifndef TNLLINEARRESIDUEGETTER_H_
#define TNLLINEARRESIDUEGETTER_H_

template< typename MatrixPointer, typename VectorPointer >
class tnlLinearResidueGetter
{
   public:

      typedef typename MatrixPointer::ObjectType MatrixType;
      typedef typename MatrixType::RealType RealType;
      typedef typename MatrixType::DeviceType DeviceType;
      typedef typename MatrixType::IndexType IndexType;

   static RealType getResidue( const MatrixPointer& matrix,
                               const VectorPointer& x,
                               const VectorPointer& b,
                               RealType bNorm = 0 );
};

#include <solvers/linear/tnlLinearResidueGetter_impl.h>

#endif /* TNLLINEARRESIDUEGETTER_H_ */
