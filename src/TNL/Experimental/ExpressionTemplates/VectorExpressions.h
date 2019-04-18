/* 
 * File: VectorExpressions.h
 * Author: Vojtěch Legler
 *
 * Created on November 29, 2018, 2:53 PM
 */

#pragma once

#include <TNL/Containers/VectorView.h>

template< typename T1, typename T2 >
class VectorAddition
{
        const T1 op1;
        const T2 op2;
        
    public:

        VectorAddition( const T1 &a, const T2 &b ): op1( a ), op2( b ){}
        
        VectorAddition( const VectorAddition &v ): op1( v.op1 ), op2( v.op2 ){}

        using RealType = typename T1::RealType;

        __cuda_callable__ 
        RealType operator[]( const int i ) const
        {
            return op1[ i ] + op2[ i ];
        }

        int getSize() const
        { 
            return op1.getSize();
        }
};

template< typename T1, typename T2 >
class VectorSubtraction
{
        const T1 op1;
        const T2 op2;
    
    public:
        
        VectorSubtraction( const T1& a, const T2& b ): op1( a ), op2( b ){}
        
        VectorSubtraction( const VectorSubtraction &v ): op1( v.op1 ), op2( v.op2 ){}

        using RealType = typename T1::RealType;

        __cuda_callable__
        RealType operator[]( const int i ) const
        {
            return op1[ i ] - op2[ i ];
        }

        int getSize() const
        { 
            return op1.getSize();
        }
};

template< typename Scalar, typename T2 >
class VectorMultiplicationLeftSide
{
        const Scalar c;
        const T2 op2;
    
    public:
        
        VectorMultiplicationLeftSide( const Scalar& a, const T2& b ): c( a ), op2( b ){}
        
        VectorMultiplicationLeftSide( const VectorMultiplicationLeftSide &v ): c( v.c ), op2( v.op2 ){}

        using RealType = typename T2::RealType;

        __cuda_callable__
        RealType operator[]( const int i ) const
        {
            return c * op2[ i ];
        }

        int getSize() const
        { 
            return op2.getSize();
        } 
};

template< typename T1 >
class VectorAbsoluteValue
{
        const T1 op1;
    
    public:
        
        VectorAbsoluteValue( const T1& a ): op1( a ){}
        
        VectorAbsoluteValue( const VectorAbsoluteValue &v ): op1( v.op1 ){}

        using RealType = typename T1::RealType;

        __cuda_callable__
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
class VectorExponentialFunction
{
        const T1 op1;
    
    public:
        
        VectorExponentialFunction( const T1& a ): op1( a ){}
        
        VectorExponentialFunction( const VectorExponentialFunction &v ): op1( v.op1 ){}

        using RealType = typename T1::RealType;

        __cuda_callable__
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
class VectorNaturalLogarithm
{
        const T1 op1;
    
    public:
        
        VectorNaturalLogarithm( const T1& a ): op1( a ){}
        
        VectorNaturalLogarithm( const VectorNaturalLogarithm &v ): op1( v.op1 ){}

        using RealType = typename T1::RealType;

        __cuda_callable__
        RealType operator[]( const int i ) const
        {
            return std::log( op1[ i ] );
        }

        int getSize() const
        { 
            return op1.getSize();
        }
};

template< typename T1 >
class VectorSine
{
        const T1 op1;
    
    public:
        
        VectorSine( const T1& a ): op1( a ){}
        
        VectorSine( const VectorSine &v ): op1( v.op1 ){}

        using RealType = typename T1::RealType;

        __cuda_callable__
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
class VectorCosine
{
        const T1 op1;
    
    public:
        
        VectorCosine( const T1& a ): op1( a ){}
        
        VectorCosine( const VectorCosine &v ): op1( v.op1 ){}

        using RealType = typename T1::RealType;

        __cuda_callable__
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
class VectorTangent
{
        const T1 op1;
    
    public:
        
        VectorTangent( const T1& a ): op1( a ){}
        
        VectorTangent( const VectorTangent &v ): op1( v.op1 ){}

        using RealType = typename T1::RealType;

        __cuda_callable__
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
class VectorArcSine
{
        const T1 op1;
    
    public:
        
        VectorArcSine( const T1& a ): op1( a ){}
        
        VectorArcSine( const VectorArcSine &v ): op1( v.op1 ){}

        using RealType = typename T1::RealType;

        __cuda_callable__
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
class VectorArcCosine
{
        const T1 op1;
    
    public:
        
        VectorArcCosine( const T1& a ): op1( a ){}
        
        VectorArcCosine( const VectorArcCosine &v ): op1( v.op1 ){}

        using RealType = typename T1::RealType;

        __cuda_callable__
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
class VectorArcTangent
{
        const T1 op1;
    
    public:
        
        VectorArcTangent( const T1& a ): op1( a ){}
        
        VectorArcTangent( const VectorArcTangent &v ): op1( v.op1 ){}

        using RealType = typename T1::RealType;

        __cuda_callable__
        RealType operator[]( const int i ) const
        {
            return std::atan( op1[ i ] );
        }

        int getSize() const
        { 
            return op1.getSize();
        }
};

template< typename T1, typename T2 >
VectorAddition< T1, T2 > operator+( const T1 &a, const T2 &b )
{
    return VectorAddition< T1, T2 >( a, b );
}

template< typename T1, typename T2 >
VectorSubtraction< T1, T2 > operator-( const T1 &a, const T2 &b )
{
    return VectorSubtraction< T1, T2 >( a, b );
}

template< typename Scalar, typename T2 >
VectorMultiplicationLeftSide< Scalar, T2 > operator*( const Scalar &a, const T2 &b )
{
    return VectorMultiplicationLeftSide< Scalar, T2 >( a, b );
}

template< typename T1 >
VectorAbsoluteValue< T1 > abs( const T1 &a )
{
    return VectorAbsoluteValue< T1 >( a );
}

template< typename T1 >
VectorExponentialFunction< T1 > exp( const T1 &a )
{
    return VectorExponentialFunction< T1 >( a );
}

template< typename T1 >
VectorNaturalLogarithm< T1 > log( const T1 &a )
{
    return VectorNaturalLogarithm< T1 >( a );
}

template< typename T1 >
VectorSine< T1 > sin( const T1 &a )
{
    return VectorSine< T1 >( a );
}

template< typename T1 >
VectorCosine< T1 > cos( const T1 &a )
{
    return VectorCosine< T1 >( a );
}

template< typename T1 >
VectorTangent< T1 > tan( const T1 &a )
{
    return VectorTangent< T1 >( a );
}

template< typename T1 >
VectorArcSine< T1 > asin( const T1 &a )
{
    return VectorArcSine< T1 >( a );
}

template< typename T1 >
VectorArcCosine< T1 > acos( const T1 &a )
{
    return VectorArcCosine< T1 >( a );
}

template< typename T1 >
VectorArcTangent< T1 > atan( const T1 &a )
{
    return VectorArcTangent< T1 >( a );
}
