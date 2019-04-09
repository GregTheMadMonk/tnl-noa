/***************************************************************************
                          ArrayOperationsStatic_impl.h  -  description
                             -------------------
    begin                : Apr 8, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/Algorithms/ArrayOperations.h>

namespace TNL {
namespace Containers {
namespace Algorithms {

template< typename Element >
__cuda_callable__
void
ArrayOperations< void >::
setElement( Element* data,
            const Element& value )
{
   *data = value;
}

template< typename Element >
__cuda_callable__
Element
ArrayOperations< void >::
getElement( const Element* data )
{
   return *data;
}

template< typename Element, typename Index >
__cuda_callable__
void
ArrayOperations< void >::
set( Element* data,
     const Element& value,
     const Index size )
{
   for( Index i = 0; i < size; i ++ )
      data[ i ] = value;
}

template< typename DestinationElement,
          typename SourceElement,
          typename Index >
__cuda_callable__
void
ArrayOperations< void >::
copy( DestinationElement* destination,
      const SourceElement* source,
      const Index size )
{
   for( Index i = 0; i < size; i ++ )
      destination[ i ] = source[ i ];
}

template< typename Element1,
          typename Element2,
          typename Index >
__cuda_callable__
bool
ArrayOperations< void >::
compare( const Element1* destination,
         const Element2* source,
         const Index size )
{
   for( Index i = 0; i < size; i++ )
      if( ! ( destination[ i ] == source[ i ] ) )
         return false;
   return true;
}

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
