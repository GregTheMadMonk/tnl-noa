/***************************************************************************
                          StaticVectorExpressions.h  -  description
                             -------------------
    begin                : Apr 19, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Containers/Expressions/BinaryExpressionTemplate.h>
#include <TNL/Containers/Expressions/ExpressionTemplatesOperations.h>
#include <TNL/Containers/Expressions/StaticComparison.h>

#include "Expressions/StaticComparison.h"

namespace TNL {
   namespace Containers {

////
// Addition
template< int Size, typename Real, typename ET >
const Expressions::BinaryExpressionTemplate< StaticVector< Size, Real >, ET, Expressions::Addition >
operator+( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::BinaryExpressionTemplate< StaticVector< Size, Real >, ET, Expressions::Addition >( a, b );
}

template< typename ET, int Size, typename Real >
const Expressions::BinaryExpressionTemplate< ET, StaticVector< Size, Real >, Expressions::Addition >
operator+( const ET& a, const StaticVector< Size, Real >& b )
{
   return Expressions::BinaryExpressionTemplate< ET, StaticVector< Size, Real >, Expressions::Addition >( a, b );
}

template< int Size, typename Real1, typename Real2 >
const Expressions::BinaryExpressionTemplate< StaticVector< Size, Real1 >, StaticVector< Size, Real2 >, Expressions::Addition >
operator+( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   return Expressions::BinaryExpressionTemplate< StaticVector< Size, Real1 >, StaticVector< Size, Real2 >, Expressions::Addition >( a, b );
}

////
// Subtraction
template< int Size, typename Real, typename ET >
const Expressions::BinaryExpressionTemplate< StaticVector< Size, Real >, ET, Expressions::Subtraction >
operator-( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::BinaryExpressionTemplate< StaticVector< Size, Real >, ET, Expressions::Subtraction >( a, b );
}

template< typename ET, int Size, typename Real >
const Expressions::BinaryExpressionTemplate< ET, StaticVector< Size, Real >, Expressions::Subtraction >
operator-( const ET& a, const StaticVector< Size, Real >& b )
{
   return Expressions::BinaryExpressionTemplate< ET, StaticVector< Size, Real >, Expressions::Subtraction >( a, b );
}

template< int Size, typename Real1, typename Real2 >
const Expressions::BinaryExpressionTemplate< StaticVector< Size, Real1 >, StaticVector< Size, Real2 >, Expressions::Subtraction >
operator-( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   return Expressions::BinaryExpressionTemplate< StaticVector< Size, Real1 >, StaticVector< Size, Real2 >, Expressions::Subtraction >( a, b );
}

////
// Multiplication
template< int Size, typename Real, typename ET >
const Expressions::BinaryExpressionTemplate< StaticVector< Size, Real >, ET, Expressions::Multiplication >
operator*( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::BinaryExpressionTemplate< StaticVector< Size, Real >, ET, Expressions::Multiplication >( a, b );
}

template< typename ET, int Size, typename Real >
const Expressions::BinaryExpressionTemplate< ET, StaticVector< Size, Real >, Expressions::Multiplication >
operator*( const ET& a, const StaticVector< Size, Real >& b )
{
   return Expressions::BinaryExpressionTemplate< ET, StaticVector< Size, Real >, Expressions::Multiplication >( a, b );
}

template< int Size, typename Real1, typename Real2 >
const Expressions::BinaryExpressionTemplate< StaticVector< Size, Real1 >, StaticVector< Size, Real2 >, Expressions::Multiplication >
operator*( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   return Expressions::BinaryExpressionTemplate< StaticVector< Size, Real1 >, StaticVector< Size, Real2 >, Expressions::Multiplication >( a, b );
}

////
// Division
template< int Size, typename Real, typename ET >
const Expressions::BinaryExpressionTemplate< StaticVector< Size, Real >, ET, Expressions::Division >
operator/( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::BinaryExpressionTemplate< StaticVector< Size, Real >, ET, Expressions::Division >( a, b );
}

template< typename ET, int Size, typename Real >
const Expressions::BinaryExpressionTemplate< ET, StaticVector< Size, Real >, Expressions::Division >
operator/( const ET& a, const StaticVector< Size, Real >& b )
{
   return Expressions::BinaryExpressionTemplate< ET, StaticVector< Size, Real >, Expressions::Division >( a, b );
}

template< int Size, typename Real1, typename Real2 >
const Expressions::BinaryExpressionTemplate< StaticVector< Size, Real1 >, StaticVector< Size, Real2 >, Expressions::Division >
operator/( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   return Expressions::BinaryExpressionTemplate< StaticVector< Size, Real1 >, StaticVector< Size, Real2 >, Expressions::Division >( a, b );
}

////
// Comparison operations - operator ==
template< int Size, typename Real, typename ET >
bool operator==( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::StaticComparisonEQ( a, b );
}

template< typename ET, int Size, typename Real >
bool operator==( const ET& a, const StaticVector< Size, Real >& b )
{
   return Expressions::StaticComparisonEQ( a, b );
}

template< int Size, typename Real1, typename Real2 >
bool operator==( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   return Expressions::StaticComparisonEQ( a, b );
}

////
// Comparison operations - operator !=
template< int Size, typename Real, typename ET >
bool operator!=( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::StaticComparisonNE( a, b );
}

template< typename ET, int Size, typename Real >
bool operator!=( const ET& a, const StaticVector< Size, Real >& b )
{
   return Expressions::StaticComparisonNE( a, b );
}

template< int Size, typename Real1, typename Real2 >
bool operator!=( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   return Expressions::StaticComparisonNE( a, b );
}

////
// Comparison operations - operator <
template< int Size, typename Real, typename ET >
bool operator<( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::StaticComparisonLT( a, b );
}

template< typename ET, int Size, typename Real >
bool operator<( const ET& a, const StaticVector< Size, Real >& b )
{
   return Expressions::StaticComparisonLT( a, b );
}

template< int Size, typename Real1, typename Real2 >
bool operator<( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   return Expressions::StaticComparisonLT( a, b );
}

////
// Comparison operations - operator <=
template< int Size, typename Real, typename ET >
bool operator<=( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::StaticComparisonLE( a, b );
}

template< typename ET, int Size, typename Real >
bool operator<=( const ET& a, const StaticVector< Size, Real >& b )
{
   return Expressions::StaticComparisonLE( a, b );
}

template< int Size, typename Real1, typename Real2 >
bool operator<=( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   return Expressions::StaticComparisonLE( a, b );
}

////
// Comparison operations - operator >
template< int Size, typename Real, typename ET >
bool operator>( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::StaticComparisonGT( a, b );
}

template< typename ET, int Size, typename Real >
bool operator>( const ET& a, const StaticVector< Size, Real >& b )
{
   return Expressions::StaticComparisonGT( a, b );
}

template< int Size, typename Real1, typename Real2 >
bool operator>( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   return Expressions::StaticComparisonGT( a, b );
}

////
// Comparison operations - operator >=
template< int Size, typename Real, typename ET >
bool operator>=( const StaticVector< Size, Real >& a, const ET& b )
{
   return Expressions::StaticComparisonGE( a, b );
}

template< typename ET, int Size, typename Real >
bool operator>=( const ET& a, const StaticVector< Size, Real >& b )
{
   return Expressions::StaticComparisonGE( a, b );
}

template< int Size, typename Real1, typename Real2 >
bool operator>=( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   return Expressions::StaticComparisonGE( a, b );
}

////
// TODO: Replace this with multiplication when its safe
template< int Size, typename Real, typename ET >
StaticVector< Size, Real >
Scale( const StaticVector< Size, Real >& a, const ET& b )
{
   StaticVector< Size, Real > result = Expressions::BinaryExpressionTemplate< StaticVector< Size, Real >, ET, Expressions::Multiplication >( a, b );
   return result;
}

template< typename ET, int Size, typename Real >
Expressions::BinaryExpressionTemplate< ET, StaticVector< Size, Real >, Expressions::Multiplication >
Scale( const ET& a, const StaticVector< Size, Real >& b )
{
   StaticVector< Size, Real > result =  Expressions::BinaryExpressionTemplate< ET, StaticVector< Size, Real >, Expressions::Multiplication >( a, b );
   return result;
}

template< int Size, typename Real1, typename Real2 >
Expressions::BinaryExpressionTemplate< StaticVector< Size, Real1 >, StaticVector< Size, Real2 >, Expressions::Multiplication >
Scale( const StaticVector< Size, Real1 >& a, const StaticVector< Size, Real2 >& b )
{
   StaticVector< Size, Real1 > result =  Expressions::BinaryExpressionTemplate< StaticVector< Size, Real1 >, StaticVector< Size, Real2 >, Expressions::Multiplication >( a, b );
   return result;
}


   } //namespace Containers
} // namespace TNL
