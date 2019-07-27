/***************************************************************************
                          TypeTraits.h  -  description
                             -------------------
    begin                : Jun 25, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <type_traits>
#include <utility>

namespace TNL {

/**
 * \brief Type trait for checking if T has getArrayData method.
 */
template< typename T >
class HasGetArrayDataMethod
{
private:
    typedef char YesType[1];
    typedef char NoType[2];

    template< typename C > static YesType& test( decltype(std::declval< C >().getArrayData()) );
    template< typename C > static NoType& test(...);

public:
    static constexpr bool value = ( sizeof( test< T >(0) ) == sizeof( YesType ) );
};

/**
 * \brief Type trait for checking if T has getSize method.
 */
template< typename T >
class HasGetSizeMethod
{
private:
    typedef char YesType[1];
    typedef char NoType[2];

    template< typename C > static YesType& test( decltype(std::declval< C >().getSize() ) );
    template< typename C > static NoType& test(...);

public:
    static constexpr bool value = ( sizeof( test< T >(0) ) == sizeof( YesType ) );
};

/**
 * \brief Type trait for checking if T has operator[] taking one index argument.
 */
template< typename T >
class HasSubscriptOperator
{
private:
   template< typename U >
   static constexpr auto check(U*)
   -> typename
      std::enable_if_t<
         ! std::is_same<
               decltype( std::declval<U>()[ std::declval<U>().getSize() ] ),
               void
            >::value,
         std::true_type
      >;

   template< typename >
   static constexpr std::false_type check(...);

   using type = decltype(check<T>(0));

public:
    static constexpr bool value = type::value;
};

/**
 * \brief Type trait for checking if T is an array type, e.g.
 *        \ref Containers::Array or \ref Containers::Vector.
 *
 * The trait combines \ref HasGetArrayDataMethod, \ref HasGetSizeMethod,
 * and \ref HasSubscriptOperator.
 */
template< typename T >
struct IsArrayType
: public std::integral_constant< bool,
            HasGetArrayDataMethod< T >::value &&
            HasGetSizeMethod< T >::value &&
            HasSubscriptOperator< T >::value >
{};

template< typename T >
struct IsStatic
{
   static constexpr bool Value = std::is_arithmetic< T >::value;
};

} //namespace TNL
