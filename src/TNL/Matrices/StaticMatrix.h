/***************************************************************************
                          StaticMatrix.h  -  description
                             -------------------
    begin                : Dec 3, 2020
    copyright            : (C) 2020 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include <TNL/Containers/NDArray.h>
#include <TNL/Containers/StaticVector.h>

namespace TNL {
namespace Matrices {

template< typename Value,
          std::size_t Rows,
          std::size_t Columns,
          typename Permutation = std::index_sequence< 0, 1 > >  // identity by default
class StaticMatrix
: public Containers::StaticNDArray< Value,
                                    Containers::SizesHolder< std::size_t, Rows, Columns >,
                                    Permutation >
{
   using Base = Containers::StaticNDArray< Value,
                                           Containers::SizesHolder< std::size_t, Rows, Columns >,
                                           Permutation >;

public:
   // inherit all assignment operators
   using Base::operator=;

   static constexpr std::size_t getRows()
   {
      return Rows;
   }

   __cuda_callable__
   static constexpr std::size_t getColumns()
   {
      return Columns;
   }

   __cuda_callable__
   Containers::StaticVector< Rows, Value >
   operator*( const Containers::StaticVector< Columns, Value > & vector ) const
   {
      Containers::StaticVector< Rows, Value > result;
      for( std::size_t i = 0; i < Rows; i++ ) {
         Value v = 0;
         for( std::size_t j = 0; j < Columns; j++ )
            v += (*this)( i, j ) * vector[ j ];
         result[ i ] = v;
      }
      return result;
   }
};

} // namespace Matrices
} // namespace TNL
