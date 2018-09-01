/***************************************************************************
                          SparseRow.h  -  description
                             -------------------
    begin                : Dec 19, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */


#pragma once

#include <type_traits>

#include <TNL/Devices/Cuda.h>

namespace TNL {
namespace Matrices {   

template< typename Real, typename Index >
class SparseRow
{
   public:

      __cuda_callable__
      SparseRow();

      __cuda_callable__
      SparseRow( Index* columns,
                          Real* values,
                          const Index length,
                          const Index step );

      __cuda_callable__
      void bind( Index* columns,
                 Real* values,
                 const Index length,
                 const Index step );

      __cuda_callable__
      void setElement( const Index& elementIndex,
                       const Index& column,
                       const Real& value );
 
      __cuda_callable__
      const Index& getElementColumn( const Index& elementIndex ) const;
 
      __cuda_callable__
      const Real& getElementValue( const Index& elementIndex ) const;
 
      void print( std::ostream& str ) const;

   protected:

	   Real* values;

	   Index* columns;

      Index length, step;
};

template< typename Real, typename Index >
std::ostream& operator << ( std::ostream& str, const SparseRow< Real, Index >& row )
{
   row.print( str );
   return str;
}

} // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/SparseRow_impl.h>
