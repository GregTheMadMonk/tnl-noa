/***************************************************************************
                          tnlTridiagonalMatrixRow.h  -  description
                             -------------------
    begin                : Dec 31, 2014
    copyright            : (C) 2014 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {

template< typename Real, typename Index >
class tnlTridiagonalMatrixRow
{
   public:

      __cuda_callable__
      tnlTridiagonalMatrixRow();

      __cuda_callable__
      tnlTridiagonalMatrixRow( Real* values,
                               const Index row,
                               const Index columns,
                               const Index step );

      __cuda_callable__
      void bind( Real* values,
                 const Index row,
                 const Index columns,
                 const Index step );

      __cuda_callable__
      void setElement( const Index& elementIndex,
                       const Index& column,
                       const Real& value );

   protected:

      Real* values;

      Index row, columns, step;
};

} // namespace TNL

#include <matrices/tnlTridiagonalMatrixRow_impl.h>
