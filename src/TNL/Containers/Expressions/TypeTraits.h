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
      HasEnabledStaticExpressionTemplates< ET1 >::value >;

template< typename ET1, typename ET2 >
using EnableIfStaticBinaryExpression_t = std::enable_if_t<
      HasEnabledStaticExpressionTemplates< ET1 >::value ||
      HasEnabledStaticExpressionTemplates< ET2 >::value >;

template< typename ET1 >
using EnableIfUnaryExpression_t = std::enable_if_t<
      HasEnabledExpressionTemplates< ET1 >::value >;

template< typename ET1, typename ET2 >
using EnableIfBinaryExpression_t = std::enable_if_t<
      // we need to avoid ambiguity with operators defined in Array (e.g. Array::operator==)
      // so the first operand must not be Array
      (
         HasAddAssignmentOperator< ET1 >::value ||
         HasEnabledExpressionTemplates< ET1 >::value ||
         std::is_arithmetic< ET1 >::value
      ) &&
      (
         HasEnabledExpressionTemplates< ET2 >::value ||
         HasEnabledExpressionTemplates< ET1 >::value
      ) >;

template< typename ET1 >
using EnableIfDistributedUnaryExpression_t = std::enable_if_t<
      HasEnabledDistributedExpressionTemplates< ET1 >::value >;

template< typename ET1, typename ET2 >
using EnableIfDistributedBinaryExpression_t = std::enable_if_t<
      // we need to avoid ambiguity with operators defined in Array (e.g. Array::operator==)
      // so the first operand must not be Array
      (
         HasAddAssignmentOperator< ET1 >::value ||
         HasEnabledDistributedExpressionTemplates< ET1 >::value ||
         std::is_arithmetic< ET1 >::value
      ) &&
      (
         HasEnabledDistributedExpressionTemplates< ET2 >::value ||
         HasEnabledDistributedExpressionTemplates< ET1 >::value
      ) >;


// helper trait class for proper classification of expression operands using getExpressionVariableType
template< typename T, typename V,
          bool enabled = IsVectorType< V >::value >
struct IsArithmeticSubtype
: public std::integral_constant< bool,
            // TODO: use std::is_assignable?
            std::is_same< T, typename V::RealType >::value >
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
template< typename R, bool enabled = ! HasEnabledStaticExpressionTemplates< R >::value >
struct RemoveExpressionTemplate
{
   using type = R;
};

template< typename R >
struct RemoveExpressionTemplate< R, false >
{
//   using type = StaticVector< R::getSize(), typename RemoveExpressionTemplate< typename R::RealType >::type >;
   using type = typename RemoveExpressionTemplate< typename R::VectorOperandType >::type;
};

template< typename R >
using RemoveET = typename RemoveExpressionTemplate< R >::type;

} // namespace Expressions
} // namespace Containers
} // namespace TNL
