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
 * \brief Type trait for checking if T has setSize method.
 */
template< typename T >
class HasSetSizeMethod
{
private:
   template< typename U >
   static constexpr auto check(U*)
   -> typename
      std::enable_if_t<
         std::is_same<
               decltype( std::declval<U>().setSize(0) ),
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
 * \brief Type trait for checking if T has operator+= taking one argument of type T.
 */
template< typename T >
class HasAddAssignmentOperator
{
private:
   template< typename U >
   static constexpr auto check(U*)
   -> typename
      std::enable_if_t<
         ! std::is_same<
               decltype( std::declval<U>() += std::declval<U>() ),
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

/**
 * \brief Type trait for checking if T is a vector type, e.g.
 *        \ref Containers::Vector or \ref Containers::VectorView.
 */
template< typename T >
struct IsVectorType
: public std::integral_constant< bool,
            IsArrayType< T >::value &&
            HasAddAssignmentOperator< T >::value >
{};

/**
 * \brief Type trait for checking if T has a \e constexpr \e getSize method.
 */
template< typename T >
struct HasConstexprGetSizeMethod
{
private:
   // implementation adopted from here: https://stackoverflow.com/a/50169108
   template< bool hasGetSize = HasGetSizeMethod< T >::value, typename = void >
   struct impl
   {
      // disable nvcc warning: invalid narrowing conversion from "unsigned int" to "int"
      // (the implementation is based on the conversion)
      #ifdef __NVCC__
         #pragma push
         #pragma diag_suppress 2361
      #elif defined(__INTEL_COMPILER)
         #pragma warning(push)
         #pragma warning(disable:3291)
      #endif
      template< typename M, M method >
      static constexpr std::true_type is_constexpr_impl( decltype(int{((*method)(), 0U)}) );
      #ifdef __NVCC__
         #pragma pop
      #elif defined(__INTEL_COMPILER)
         // FIXME: this does not work - warning would be shown again...
         //#pragma warning(pop)
      #endif

      template< typename M, M method >
      static constexpr std::false_type is_constexpr_impl(...);

      using type = decltype(is_constexpr_impl< decltype(&T::getSize), &T::getSize >(0));
   };

   // specialization for types which don't have getSize() method at all
   template< typename _ >
   struct impl< false, _ >
   {
      using type = std::false_type;
   };

   using type = typename impl<>::type;

public:
   static constexpr bool value = type::value;
};

/**
 * \brief Type trait for checking if T is a static array type.
 *
 * Static array types are array types which have a \e constexpr \e getSize
 * method.
 */
template< typename T >
struct IsStaticArrayType
: public std::integral_constant< bool,
            HasConstexprGetSizeMethod< T >::value &&
            HasSubscriptOperator< T >::value >
{};

/**
 * \brief Type trait for checking if T is a view type.
 */
template< typename T >
struct IsViewType
: public std::integral_constant< bool,
            std::is_same< typename T::ViewType, T >::value >
{};

/**
 * \brief Type trait for checking if T has a static getSerializationType method.
 */
template< typename T >
class HasStaticGetSerializationType
{
private:
   template< typename U >
   static constexpr auto check(U*)
   -> typename
      std::enable_if_t<
         ! std::is_same<
               decltype( U::getSerializationType() ),
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

} //namespace TNL
