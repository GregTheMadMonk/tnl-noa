/***************************************************************************
                          tnlMultidiagonalMatrixRow.h  -  description
                             -------------------
    begin                : Jan 2, 2015
    copyright            : (C) 2015 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {

template< typename Real, typename Index >
class tnlMultidiagonalMatrixRow
{
   public:

      __cuda_callable__
      tnlMultidiagonalMatrixRow();

      __cuda_callable__
      tnlMultidiagonalMatrixRow( Real* values,
                                 Index* diagonals,
                                 const Index maxRowLength,
                                 const Index row,
                                 const Index columns,
                                 const Index step );

      __cuda_callable__
      void bind( Real* values,
                 Index* diagonals,
                 const Index maxRowLength,
                 const Index row,
                 const Index columns,
                 const Index step );

      __cuda_callable__
      void setElement( const Index& elementIndex,
                       const Index& column,
                       const Real& value );

   protected:

      Real* values;

      Index* diagonals;

      Index row, columns, maxRowLength, step;
};

} // namespace TNL

#include <TNL/matrices/tnlMultidiagonalMatrixRow_impl.h>

