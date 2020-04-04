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
   namespace Containers {
      namespace Segments {
         namespace details {

template< typename Index,
          typename Lambda >
class CheckFetchLambdaAcceptsSegmentIdxAndCompute
{
   private:
       typedef char YesType[1];
       typedef char NoType[2];

       template< typename C > static YesType& test( decltype(std::declval< C >()( Index(), Index(), Index(), std::declval< bool& >() ) ) );
       template< typename C > static NoType& test(...);

   public:
       static constexpr bool value = ( sizeof( test< Lambda >(0) ) == sizeof( YesType ) );
};

template< typename Index,
          typename Lambda >
class CheckFetchLambdaAcceptsSegmentIdx
{
   private:
       typedef char YesType[1];
       typedef char NoType[2];

       template< typename C > static YesType& test( decltype(std::declval< C >()( Index(), Index(), Index() ) ) );
       template< typename C > static NoType& test(...);

   public:
       static constexpr bool value = ( sizeof( test< Lambda >(0) ) == sizeof( YesType ) );
};

template< typename Index,
          typename Lambda >
class CheckFetchLambdaAcceptsCompute
{
   private:
       typedef char YesType[1];
       typedef char NoType[2];

       template< typename C > static YesType& test( decltype(std::declval< C >()( Index(), Index(), std::declval< bool& >() ) ) );
       template< typename C > static NoType& test(...);

   public:
       static constexpr bool value = ( sizeof( test< Lambda >(0) ) == sizeof( YesType ) );
};


template< typename Index,
          typename Lambda >
class CheckFetchLambda
{
   static constexpr bool AcceptsSegmentIdxAndCompute = CheckFetchLambdaAcceptsSegmentIdxAndCompute< Index, Lambda >::value;
   static constexpr bool AcceptsSegmentIdx = CheckFetchLambdaAcceptsSegmentIdx< Index, Lambda >::value;
   static constexpr bool AcceptsCompute = CheckFetchLambdaAcceptsCompute< Index, Lambda >::value;

   public:
      static constexpr bool acceptsSegmentIdx() { return AcceptsSegmentIdxAndCompute || AcceptsSegmentIdx; };
      static constexpr bool acceptsCompute() { return AcceptsSegmentIdxAndCompute || AcceptsCompute; };
};

         } // namespace details
      } // namespace Segements
   }  // namespace Conatiners
} // namespace TNL
