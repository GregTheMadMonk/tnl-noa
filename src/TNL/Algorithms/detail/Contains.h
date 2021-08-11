/***************************************************************************
                          Contains.h  -  description
                             -------------------
    begin                : Jul 27, 2021
    copyright            : (C) 2021 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Devices/Sequential.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Cuda/CudaCallable.h>
#include <TNL/Algorithms/reduce.h>

namespace TNL {
namespace Algorithms {
namespace detail {

template< typename Device >
struct Contains;

template< typename Device >
struct ContainsOnlyValue;


template<>
struct Contains< Devices::Sequential >
{
   template< typename Element,
             typename Index >
   __cuda_callable__
   bool operator()( const Element* data,
                    const Index size,
                    const Element& value )
   {
      if( size == 0 ) return false;
      TNL_ASSERT_TRUE( data, "Attempted to check data through a nullptr." );
      TNL_ASSERT_GE( size, 0, "" );

      for( Index i = 0; i < size; i++ )
         if( data[ i ] == value )
            return true;
      return false;
   }
};

template<>
struct ContainsOnlyValue< Devices::Sequential >
{
   template< typename Element,
             typename Index >
   __cuda_callable__
   bool operator()( const Element* data,
                    const Index size,
                    const Element& value )
   {
      if( size == 0 ) return false;
      TNL_ASSERT_TRUE( data, "Attempted to check data through a nullptr." );
      TNL_ASSERT_GE( size, 0, "" );

      for( Index i = 0; i < size; i++ )
         if( ! ( data[ i ] == value ) )
            return false;
      return true;
   }
};


template<>
struct Contains< Devices::Host >
{
   template< typename Element,
             typename Index >
   bool operator()( const Element* data,
                    const Index size,
                    const Element& value )
   {
      if( size == 0 ) return false;
      TNL_ASSERT_TRUE( data, "Attempted to check data through a nullptr." );
      TNL_ASSERT_GE( size, 0, "" );

      if( Devices::Host::isOMPEnabled() && Devices::Host::getMaxThreadsCount() > 1 ) {
         auto fetch = [=] ( Index i ) -> bool { return data[ i ] == value; };
         return reduce< Devices::Host >( ( Index ) 0, size, fetch, std::logical_or<>{}, false );
      }
      else {
         // sequential algorithm can return as soon as it finds a match
         return Contains< Devices::Sequential >{}( data, size, value );
      }
   }
};

template<>
struct ContainsOnlyValue< Devices::Host >
{
   template< typename Element,
             typename Index >
   bool operator()( const Element* data,
                    const Index size,
                    const Element& value )
   {
      if( size == 0 ) return false;
      TNL_ASSERT_TRUE( data, "Attempted to check data through a nullptr." );
      TNL_ASSERT_GE( size, 0, "" );

      if( Devices::Host::isOMPEnabled() && Devices::Host::getMaxThreadsCount() > 1 ) {
         auto fetch = [data, value] ( Index i ) -> bool { return data[ i ] == value; };
         return reduce< Devices::Host >( ( Index ) 0, size, fetch, std::logical_and<>{}, true );
      }
      else {
         // sequential algorithm can return as soon as it finds a mismatch
         return ContainsOnlyValue< Devices::Sequential >{}( data, size, value );
      }
   }
};


template<>
struct Contains< Devices::Cuda >
{
   template< typename Element,
             typename Index >
   bool operator()( const Element* data,
                    const Index size,
                    const Element& value )
   {
      if( size == 0 ) return false;
      TNL_ASSERT_TRUE( data, "Attempted to check data through a nullptr." );
      TNL_ASSERT_GE( size, (Index) 0, "" );

      auto fetch = [=] __cuda_callable__ ( Index i ) -> bool { return data[ i ] == value; };
      return reduce< Devices::Cuda >( ( Index ) 0, size, fetch, std::logical_or<>{}, false );
   }
};

template<>
struct ContainsOnlyValue< Devices::Cuda >
{
   template< typename Element,
             typename Index >
   bool operator()( const Element* data,
                    const Index size,
                    const Element& value )
   {
      if( size == 0 ) return false;
      TNL_ASSERT_TRUE( data, "Attempted to check data through a nullptr." );
      TNL_ASSERT_GE( size, 0, "" );

      auto fetch = [=] __cuda_callable__ ( Index i ) -> bool { return data[ i ] == value; };
      return reduce< Devices::Cuda >( ( Index ) 0, size, fetch, std::logical_and<>{}, true );
   }
};

} // namespace detail
} // namespace Algorithms
} // namespace TNL
