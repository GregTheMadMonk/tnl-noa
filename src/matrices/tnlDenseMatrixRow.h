/***************************************************************************
                          tnlDenseMatrixRow.h  -  description
                             -------------------
    begin                : Dec 24, 2014
    copyright            : (C) 2014 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLDENSEMATRIXROW_H_
#define TNLDENSEMATRIXROW_H_

template< typename Real, typename Index >
class tnlDenseMatrixRow
{
   public:

      __cuda_callable__
      tnlDenseMatrixRow();

      __cuda_callable__
      tnlDenseMatrixRow( Real* values,
                         const Index columns,
                         const Index step );

      __cuda_callable__
      void bind( Real* values,
                 const Index columns,
                 const Index step );

      __cuda_callable__
      void setElement( const Index& elementIndex,
                       const Index& column,
                       const Real& value );

   protected:

      Real* values;

      Index columns, step;
};

#include <matrices/tnlDenseMatrixRow_impl.h>


#endif /* TNLDENSEMATRIXROW_H_ */
