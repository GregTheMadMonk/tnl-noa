/***************************************************************************
                          LambdaAdapter.h -  description
                             -------------------
    begin                : Dpr 4, 2020
    copyright            : (C) 2020 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include "CheckLambdas.h"

namespace TNL {
   namespace Algorithms {
      namespace Segments {
         namespace detail {

template< typename Index,
          typename Lambda,
          bool AllParameters = CheckFetchLambda< Index, Lambda >::hasAllParameters() >
struct FetchLambdaAdapter
{
};

template< typename Index,
          typename Lambda >
struct FetchLambdaAdapter< Index, Lambda, true >
{
   using ReturnType = decltype( std::declval< Lambda >()( Index(), Index(), Index(), std::declval< bool& >() ) );

   __cuda_callable__
   static ReturnType call( Lambda& f, Index segmentIdx, Index localIdx, Index globalIdx, bool& compute )
   {
      return f( segmentIdx, localIdx, globalIdx, compute );
   }
};

template< typename Index,
          typename Lambda >
struct FetchLambdaAdapter< Index, Lambda, false >
{
   using ReturnType = decltype( std::declval< Lambda >()( Index(), std::declval< bool& >() ) );

   __cuda_callable__
   static ReturnType call( Lambda& f, Index segmentIdx, Index localIdx, Index globalIdx, bool& compute )
   {
      return f( globalIdx, compute );
   }
};

         } // namespace detail
      } // namespace Segments
   }  // namespace Algorithms
} // namespace TNL
