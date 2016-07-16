/***************************************************************************
                          tnlLinearResidueGetter.h  -  description
                             -------------------
    begin                : Nov 25, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLLINEARRESIDUEGETTER_H_
#define TNLLINEARRESIDUEGETTER_H_

template< typename Matrix, typename Vector >
class tnlLinearResidueGetter
{
   public:

   typedef typename Matrix :: RealType RealType;
   typedef typename Matrix :: DeviceType DeviceType;
   typedef typename Matrix :: IndexType IndexType;

   static RealType getResidue( const Matrix& matrix,
                               const Vector& x,
                               const Vector& b,
                               RealType bNorm = 0 );
};

#include <solvers/linear/tnlLinearResidueGetter_impl.h>

#endif /* TNLLINEARRESIDUEGETTER_H_ */
