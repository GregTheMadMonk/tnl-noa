/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   StaticVectorExpressions.h
 * Author: oberhuber
 *
 * Created on November 29, 2018, 2:53 PM
 */

#pragma once
#include <TNL/Containers/StaticVector.h>
#include <TNL/Experimental/ExpressionTemplates/ExpressionTemplatesOperations.h>
#include <TNL/Experimental/ExpressionTemplates/ExpressionVariableType.h>

namespace TNL {


template< typename T1,
          typename T2,
          template< typename, typename > class Operation,
          ExpressionVariableType T1Type = ExpressionVariableTypeGetter< T1 >::value,
          ExpressionVariableType T2Type = ExpressionVariableTypeGetter< T2 >::value >
struct BinaryExpressionTemplate
{
   BinaryExpressionTemplate( const T1& a, const T2& b ){};

   static T1 evaluate( const T1& a, const T2& b )
   {
      return Operation< T1, T2 >::evaluate( a, b );
   }
};

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
struct BinaryExpressionTemplate< T1, T2, Operation, VectorVariable, VectorVariable >
{
   using RealType = typename T1::RealType;
   using IsExpressionTemplate = bool;

   BinaryExpressionTemplate( const T1& a, const T2& b ): op1( a ), op2( b ){}

   static BinaryExpressionTemplate evaluate( const T1& a, const T2& b )
   {
      return BinaryExpressionTemplate( a, b );
   }

   RealType operator[]( const int i ) const
   {
       return Operation< typename T1::RealType, typename T2::RealType >::evaluate( op1[ i ], op2[ i ] );
   }

   int getSize() const
   {
       return op1.getSize();
   }

   protected:
      const T1 &op1;
      const T2 &op2;

};

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
struct BinaryExpressionTemplate< T1, T2, Operation, VectorVariable, ArithmeticVariable  >
{
   using RealType = typename T1::RealType;
   using IsExpressionTemplate = bool;

   BinaryExpressionTemplate( const T1& a, const T2& b ): op1( a ), op2( b ){}

   BinaryExpressionTemplate evaluate( const T1& a, const T2& b )
   {
      return BinaryExpressionTemplate( a, b );
   }

   RealType operator[]( const int i ) const
   {
       return Operation< typename T1::RealType, T2 >::evaluate( op1[ i ], op2 );
   }

   int getSize() const
   {
       return op1.getSize();
   }

   protected:
      const T1 &op1;
      const T2 &op2;

};

template< typename T1,
          typename T2,
          template< typename, typename > class Operation >
struct BinaryExpressionTemplate< T1, T2, Operation, ArithmeticVariable, VectorVariable  >
{
   using RealType = typename T2::RealType;
   using IsExpressionTemplate = bool;

   BinaryExpressionTemplate( const T1& a, const T2& b ): op1( a ), op2( b ){}

   BinaryExpressionTemplate evaluate( const T1& a, const T2& b )
   {
      return BinaryExpressionTemplate( a, b );
   }

   RealType operator[]( const int i ) const
   {
       return Operation< T1, typename T2::RealType >::evaluate( op1, op2[ i ] );
   }

   int getSize() const
   {
       return op2.getSize();
   }

   protected:
      const T1& op1;
      const T2& op2;

};


template< typename T1, typename T2 >
BinaryExpressionTemplate< T1, T2, ExpressionTemplates::Addition > operator+( const T1 &a, const T2 &b )
{
    return BinaryExpressionTemplate< T1, T2, ExpressionTemplates::Addition >( a, b );
}

template< typename T1, typename T2 >
BinaryExpressionTemplate< T1, T2, ExpressionTemplates::Subtraction > operator-( const T1 &a, const T2 &b )
{
    return BinaryExpressionTemplate< T1, T2, ExpressionTemplates::Subtraction >( a, b );
}

template< typename T1, typename T2 >
BinaryExpressionTemplate< T1, T2, ExpressionTemplates::Multiplication > operator*( const T1 &a, const T2 &b )
{
    return BinaryExpressionTemplate< T1, T2, ExpressionTemplates::Multiplication >( a, b );
}

template< typename T1, typename T2 >
BinaryExpressionTemplate< T1, T2, ExpressionTemplates::Division > operator/( const T1 &a, const T2 &b )
{
    return BinaryExpressionTemplate< T1, T2, ExpressionTemplates::Division >( a, b );
}



template< typename T1 >
class StaticVectorAbsoluteValue
{
        const T1 &op1;

    public:

        StaticVectorAbsoluteValue( const T1& a ): op1( a ){}

        using RealType = typename T1::RealType;

        RealType operator[]( const int i ) const
        {
            return std::abs( op1[ i ] );
        }

        int getSize() const
        {
            return op1.getSize();
        }
};

template< typename T1 >
class StaticVectorExponentialFunction
{
        const T1 &op1;

    public:

        StaticVectorExponentialFunction( const T1& a ): op1( a ){}

        using RealType = typename T1::RealType;

        RealType operator[]( const int i ) const
        {
            return std::exp( op1[ i ] );
        }

        int getSize() const
        {
            return op1.getSize();
        }
};

template< typename T1 >
class StaticVectorNaturalLogarithm
{
        const T1 &op1;

    public:

        StaticVectorNaturalLogarithm( const T1& a ): op1( a ){}

        using RealType = typename T1::RealType;

        RealType operator[]( const int i ) const
        {
            return std::log( op1[ i ] );
        }

        int getSize() const
        {
            return op1.getSize();
        }
};
/*
template< typename T1 >
class StaticVectorSquareRoot
{
        const T1 &op1;

    public:

        StaticVectorSquareRoot( const T1& a ): op1( a ){}

        using RealType = typename T1::RealType;

        RealType operator[]( const int i ) const
        {
            return std::sqrt( op1[ i ] );
        }

        int getSize() const
        {
            return op1.getSize();
        }
};*/

template< typename T1 >
class StaticVectorSine
{
        const T1 &op1;

    public:

        StaticVectorSine( const T1& a ): op1( a ){}

        using RealType = typename T1::RealType;

        RealType operator[]( const int i ) const
        {
            return std::sin( op1[ i ] );
        }

        int getSize() const
        {
            return op1.getSize();
        }
};

template< typename T1 >
class StaticVectorCosine
{
        const T1 &op1;

    public:

        StaticVectorCosine( const T1& a ): op1( a ){}

        using RealType = typename T1::RealType;

        RealType operator[]( const int i ) const
        {
            return std::cos( op1[ i ] );
        }

        int getSize() const
        {
            return op1.getSize();
        }
};

template< typename T1 >
class StaticVectorTangent
{
        const T1 &op1;

    public:

        StaticVectorTangent( const T1& a ): op1( a ){}

        using RealType = typename T1::RealType;

        RealType operator[]( const int i ) const
        {
            return std::tan( op1[ i ] );
        }

        int getSize() const
        {
            return op1.getSize();
        }
};

template< typename T1 >
class StaticVectorArcSine
{
        const T1 &op1;

    public:

        StaticVectorArcSine( const T1& a ): op1( a ){}

        using RealType = typename T1::RealType;

        RealType operator[]( const int i ) const
        {
            return std::asin( op1[ i ] );
        }

        int getSize() const
        {
            return op1.getSize();
        }
};

template< typename T1 >
class StaticVectorArcCosine
{
        const T1 &op1;

    public:

        StaticVectorArcCosine( const T1& a ): op1( a ){}

        using RealType = typename T1::RealType;

        RealType operator[]( const int i ) const
        {
            return std::acos( op1[ i ] );
        }

        int getSize() const
        {
            return op1.getSize();
        }
};

template< typename T1 >
class StaticVectorArcTangent
{
        const T1 &op1;

    public:

        StaticVectorArcTangent( const T1& a ): op1( a ){}

        using RealType = typename T1::RealType;

        RealType operator[]( const int i ) const
        {
            return std::atan( op1[ i ] );
        }

        int getSize() const
        {
            return op1.getSize();
        }
};


template< typename T1 >
StaticVectorAbsoluteValue< T1 > abs( const T1 &a )
{
    return StaticVectorAbsoluteValue< T1 >( a );
}

template< typename T1 >
StaticVectorExponentialFunction< T1 > exp( const T1 &a )
{
    return StaticVectorExponentialFunction< T1 >( a );
}

template< typename T1 >
StaticVectorNaturalLogarithm< T1 > log( const T1 &a )
{
    return StaticVectorNaturalLogarithm< T1 >( a );
}
/*
template< typename T1 >
StaticVectorSquareRoot< T1 > sqrt( const T1 &a )
{
    return StaticVectorSquareRoot< T1 >( a );
}*/

/*
template< typename T1 >
StaticVectorSine< T1 > sin( const T1 &a )
{
    return StaticVectorSine< T1 >( a );
}

template< typename T1 >
StaticVectorCosine< T1 > cos( const T1 &a )
{
    return StaticVectorCosine< T1 >( a );
}

template< typename T1 >
StaticVectorTangent< T1 > tan( const T1 &a )
{
    return StaticVectorTangent< T1 >( a );
}

template< typename T1 >
StaticVectorArcSine< T1 > asin( const T1 &a )
{
    return StaticVectorArcSine< T1 >( a );
}

template< typename T1 >
StaticVectorArcCosine< T1 > acos( const T1 &a )
{
    return StaticVectorArcCosine< T1 >( a );
}

template< typename T1 >
StaticVectorArcTangent< T1 > atan( const T1 &a )
{
    return StaticVectorArcTangent< T1 >( a );
}*/

} //namespace TNL
