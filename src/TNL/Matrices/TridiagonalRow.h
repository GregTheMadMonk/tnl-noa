/***************************************************************************
                          TridiagonalRow.h  -  description
                             -------------------
    begin                : Dec 31, 2014
    copyright            : (C) 2014 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Matrices {   

template< typename Real, typename Index >
class TridiagonalRow
{
   public:

      __cuda_callable__
      TridiagonalRow();

      __cuda_callable__
      TridiagonalRow( Real* values,
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

} // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/TridiagonalRow_impl.h>
