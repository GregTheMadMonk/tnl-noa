/***************************************************************************
                          TypeTraits.h  -  description
                             -------------------
    begin                : Jul 26, 2019
    copyright            : (C) 2019 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/TypeTraits.h>

namespace TNL {
namespace Containers {
namespace Expressions {

// trait classes used for the enabling of expression templates
template< typename T >
struct HasEnabledExpressionTemplates : std::false_type
{};

template< typename T >
struct HasEnabledStaticExpressionTemplates : std::false_type
{};

template< typename T >
struct HasEnabledDistributedExpressionTemplates : std::false_type
{};


// type aliases for enabling specific operators and functions using SFINAE
template< typename ET1 >
using EnableIfStaticUnaryExpression_t = std::enable_if_t<
      HasEnabledStaticExpressionTemplates< std::decay_t< ET1 > >::value >;

template< typename ET1, typename ET2 >
using EnableIfStaticBinaryExpression_t = std::enable_if_t<
      (
         HasEnabledStaticExpressionTemplates< std::decay_t< ET1 > >::value ||
         HasEnabledStaticExpressionTemplates< std::decay_t< ET2 > >::value
      ) && !
      (
         HasEnabledExpressionTemplates< std::decay_t< ET2 > >::value ||
         HasEnabledExpressionTemplates< std::decay_t< ET1 > >::value ||
         HasEnabledDistributedExpressionTemplates< std::decay_t< ET2 > >::value ||
         HasEnabledDistributedExpressionTemplates< std::decay_t< ET1 > >::value
      ) >;

template< typename ET1 >
using EnableIfUnaryExpression_t = std::enable_if_t<
      HasEnabledExpressionTemplates< std::decay_t< ET1 > >::value >;

template< typename ET1, typename ET2 >
using EnableIfBinaryExpression_t = std::enable_if_t<
      // we need to avoid ambiguity with operators defined in Array (e.g. Array::operator==)
      // so the first operand must not be Array
      (
         HasAddAssignmentOperator< std::decay_t< ET1 > >::value ||
         HasEnabledExpressionTemplates< std::decay_t< ET1 > >::value ||
         std::is_arithmetic< std::decay_t< ET1 > >::value
      ) &&
      (
         HasEnabledExpressionTemplates< std::decay_t< ET2 > >::value ||
         HasEnabledExpressionTemplates< std::decay_t< ET1 > >::value
      ) >;

template< typename ET1 >
using EnableIfDistributedUnaryExpression_t = std::enable_if_t<
      HasEnabledDistributedExpressionTemplates< std::decay_t< ET1 > >::value >;

template< typename ET1, typename ET2 >
using EnableIfDistributedBinaryExpression_t = std::enable_if_t<
      // we need to avoid ambiguity with operators defined in Array (e.g. Array::operator==)
      // so the first operand must not be Array
      (
         HasAddAssignmentOperator< std::decay_t< ET1 > >::value ||
         HasEnabledDistributedExpressionTemplates< std::decay_t< ET1 > >::value ||
         std::is_arithmetic< std::decay_t< ET1 > >::value
      ) &&
      (
         HasEnabledDistributedExpressionTemplates< std::decay_t< ET2 > >::value ||
         HasEnabledDistributedExpressionTemplates< std::decay_t< ET1 > >::value
      ) >;


// helper trait class for proper classification of expression operands using getExpressionVariableType
template< typename T, typename V,
          bool enabled = IsVectorType< V >::value >
struct IsArithmeticSubtype
: public std::integral_constant< bool,
            // TODO: use std::is_assignable?
            std::is_same< T, typename std::decay_t< V >::RealType >::value >
{};

template< typename T >
struct IsArithmeticSubtype< T, T, true >
: public std::false_type
{};

template< typename T >
struct IsArithmeticSubtype< T, T, false >
: public std::false_type
{};

template< typename T, typename V >
struct IsArithmeticSubtype< T, V, false >
: public std::is_arithmetic< T >
{};


// helper trait class (used in unit tests)
template<class T, class R = void>
struct enable_if_type { typedef R type; };

template< typename R, typename Enable = void >
struct RemoveExpressionTemplate
{
   using type = std::decay_t< R >;
};

template< typename R >
struct RemoveExpressionTemplate< R, typename enable_if_type< typename std::decay_t< R >::VectorOperandType >::type >
{
   using type = typename RemoveExpressionTemplate< typename std::decay_t< R >::VectorOperandType >::type;
};

template< typename R >
using RemoveET = typename RemoveExpressionTemplate< R >::type;

// helper trait class for Static*ExpressionTemplates classes
template< typename R, typename Enable = void >
struct OperandMemberType
{
   using type = std::conditional_t< std::is_fundamental< R >::value,
                     // non-reference for fundamental types
                     std::add_const_t< std::remove_reference_t< R > >,
                     // lvalue-reference for other types (especially StaticVector)
                     std::add_lvalue_reference_t< std::add_const_t< R > >
                  >;
//   using type = std::add_const_t< std::remove_reference_t< R > >;
};

// assuming that only the StaticBinaryExpressionTemplate and StaticUnaryTemplate classes have a VectorOperandType type member
template< typename R >
struct OperandMemberType< R, typename enable_if_type< typename R::VectorOperandType >::type >
{
   // non-reference for StaticBinaryExpressionTemplate and StaticUnaryExpressionTemplate
   // (otherwise we would get segfaults - binding const-reference to temporary Static*ExpressionTemplate
   // objects does not work as expected...)
   using type = std::add_const_t< std::remove_reference_t< R > >;
};

} // namespace Expressions
} // namespace Containers
} // namespace TNL
