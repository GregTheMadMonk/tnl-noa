/***************************************************************************
                          DenseRow.h  -  description
                             -------------------
    begin                : Dec 24, 2014
    copyright            : (C) 2014 by oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Matrices {   

template< typename Real, typename Index >
class DenseRow
{
   public:

      __cuda_callable__
      DenseRow();

      __cuda_callable__
      DenseRow( Real* values,
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

} // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/DenseRow_impl.h>

