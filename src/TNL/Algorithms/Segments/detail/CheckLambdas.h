/***************************************************************************
                          CheckLambdas.h -  description
                             -------------------
    begin                : Dpr 4, 2020
    copyright            : (C) 2020 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once


namespace TNL {
   namespace Algorithms {
      namespace Segments {
         namespace detail {

template< typename Index,
          typename Lambda >
class CheckFetchLambda
{
   private:
      typedef char YesType[1];
      typedef char NoType[2];

      template< typename C > static YesType& test( decltype(std::declval< C >()( Index(), Index(), Index(), std::declval< bool& >() ) ) );
      template< typename C > static NoType& test(...);

      static constexpr bool value = ( sizeof( test< Lambda >(0) ) == sizeof( YesType ) );

   public:

      static constexpr bool hasAllParameters() { return value; };
};

         } // namespace detail
      } // namespace Segments
   }  // namespace Algorithms
} // namespace TNL
