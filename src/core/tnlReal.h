/***************************************************************************
                          tnlReal.h
                             -------------------
    begin                : Jun 14, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */



#ifndef TNLREAL_H_
#define TNLREAL_H_

#include <iostream>
#include <math.h>
#include <core/tnlFlopsCounter.h>

template< class T > class tnlReal
{
   T data;

   public:

   tnlReal()
   : data( 0 )
     {};

   template< class S >tnlReal( const S& d )
   : data( d )
     {}

   tnlReal( const tnlReal& v )
   : data( v. data )
     {};

   T& Data()
   {
      return data;
   };

   const T& Data() const
   {
      return data;
   };

   const tnlReal& operator += ( const tnlReal& v )
   {
      data += v. data;
      tnl_flops_counter. recordAdding();
      return *this;
   };

	template< class S > const tnlReal& operator += ( const S& v )
	{
		data += v;
		tnl_flops_counter. recordAdding();
		return *this;
	}

	const tnlReal& operator -= ( const tnlReal& v )
	{
		data -= v. data;
		tnl_flops_counter. recordAdding();
		return *this;
	};

	template< class S > const tnlReal& operator -= ( const S& v )
	{
		data -= v;
		tnl_flops_counter. recordAdding();
		return *this;
	}

	const tnlReal& operator *= ( const tnlReal& v )
	{
		data *= v. data;
		tnl_flops_counter. recordMultiplying();
		return *this;
	};

	template< class S > const tnlReal& operator *= ( const S& v )
	{
		data *= v;
		tnl_flops_counter. recordMultiplying();
		return *this;
	}

	const tnlReal& operator /= ( const tnlReal& v )
	{
		data /= v. data;
		tnl_flops_counter. recordDividing();
		return *this;
	};

	template< class S > const tnlReal& operator /= ( const S& v )
	{
		data /= v;
		tnl_flops_counter. recordDividing();
		return *this;
	}

	const tnlReal& operator + ( const tnlReal& v ) const
	{
		return tnlReal( *this ) += v;
	};

	template< class S > const tnlReal& operator + ( const S& v ) const
	{
		return tnlReal( *this ) += v;
	}

	const tnlReal& operator - ( const tnlReal& v ) const
	{
		return tnlReal( *this ) -= v;
	};

	template< class S > const tnlReal& operator - ( const S& v ) const
	{
		return tnlReal( *this ) -= v;
	}

	const tnlReal& operator - () const
	{
		return tnlReal( *this ) *= -1.0;
	};

	const tnlReal& operator * ( const tnlReal& v ) const
	{
		return tnlReal( *this ) *= v;
	};

	template< class S > const tnlReal& operator * ( const S& v ) const
	{
		return tnlReal( *this ) *= v;
	}

	const tnlReal& operator / ( const tnlReal& v ) const
	{
		return tnlReal( *this ) /= v;
	};

	template< class S > const tnlReal& operator / ( const S& v ) const
	{
		return tnlReal( *this ) /= v;
	}

	const tnlReal& operator = ( const tnlReal& v )
	{
		data = v. data;
		return *this;
	};

	const tnlReal& operator = ( const T& v )
	{
		data = v;
		return *this;
	};

	bool operator == ( const tnlReal& v ) const
	{
		return data == v.data;
	};

	bool operator == ( const T& v ) const
	{
		return data == v;
	};

	bool operator != ( const T& v ) const
	{
		return data != v;
	};

	bool operator != ( const tnlReal& v ) const
	{
		return data != v.data;
	};

	bool operator <= ( const T& v ) const
	{
		return data <= v;
	};

	bool operator <= ( const tnlReal& v ) const
	{
		return data <= v.data;
	};

	bool operator >= ( const T& v ) const
	{
		return data >= v;
	};

	bool operator >= ( const tnlReal& v ) const
	{
		return data >= v.data;
	};

	bool operator < ( const T& v ) const
	{
		return data < v;
	};

	bool operator < ( const tnlReal& v ) const
	{
		return data < v.data;
	};

	bool operator > ( const T& v ) const
	{
		return data > v;
	};

	bool operator > ( const tnlReal& v ) const
	{
		return data > v.data;
	};

	bool operator || ( const tnlReal& v ) const
    {
		return data || v.data;
    };

	bool operator && ( const tnlReal& v ) const
    {
		return data && v.data;
    };

	bool operator ! () const
   {
	   return ! data;
   };

    /*operator bool () const
    {
	   return ( bool ) data;
	};*/

	operator int () const
	{
	   return ( int ) data;
	};

	/*operator float () const
	{
		return ( float ) data;
	};*/

	operator double () const
    {
		return ( double ) data;
    };

};

template< class T, class S > const tnlReal< T >& operator + ( const S& v1, const tnlReal< T >& v2 )
{
   return tnlReal< T >( v1 ) += v2. Data();
};

template< class T, class S > const tnlReal< T >& operator - ( const S& v1, const tnlReal< T >& v2 )
{
   return tnlReal< T >( v1 ) -= v2. Data();
};

template< class T, class S > const tnlReal< T >& operator * ( const S& v1, const tnlReal< T >& v2 )
{
   return tnlReal< T >( v1 ) *= v2. Data();
};

template< class T, class S > const tnlReal< T >& operator / ( const S& v1, const tnlReal< T >& v2 )
{
   return tnlReal< T >( v1 ) /= v2. Data();
};

template< class T > bool operator == ( const T& v1, const tnlReal< T >& v2 )
{
   return v1 == v2. Data();
};

template< class T > bool operator != ( const T& v1, const tnlReal< T >& v2 )
{
   return v1 != v2. Data();
};

template< class T > bool operator <= ( const T& v1, const tnlReal< T >& v2 )
{
   return v1 <= v2. Data();
};

template< class T > bool operator >= ( const T& v1, const tnlReal< T >& v2 )
{
   return v1 >= v2. Data();
};

template< class T > bool operator < ( const T& v1, const tnlReal< T >& v2 )
{
   return v1 < v2. Data();
};

template< class T > bool operator > ( const T& v1, const tnlReal< T >& v2 )
{
   return v1 > v2. Data();
};

template< class T > const tnlReal< T > fabs( const tnlReal< T >& v )
{
   tnl_flops_counter. recordFunction();
   return tnlReal< T >( fabs( v. Data() ) );
};

template< class T > const tnlReal< T > sqrt( const tnlReal< T >& v )
{
   tnl_flops_counter. recordFunction();
   return tnlReal< T >( sqrt( v. Data() ) );
};

template< class T > const tnlReal< T > pow( const tnlReal< T >& x, const tnlReal< T >& exp )
{
   tnl_flops_counter. recordFunction();
   return tnlReal< T >( pow( x. Data(), exp. Data() ) );
};

template< class T > const tnlReal< T > pow( const tnlReal< T >& x, const T& exp )
{
   tnl_flops_counter. recordFunction();
   return tnlReal< T >( pow( x. Data(), exp ) );
};

template< class T > const tnlReal< T > cos( const tnlReal< T >& x )
{
   tnl_flops_counter. recordFunction();
   return tnlReal< T >( cos( x. Data() ) );
};

template< class T > const tnlReal< T > sin( const tnlReal< T >& x )
{
   tnl_flops_counter. recordFunction();
   return tnlReal< T >( sin( x. Data() ) );
};

template< class T > const tnlReal< T > tan( const tnlReal< T >& x )
{
   tnl_flops_counter. recordFunction();
   return tnlReal< T >( tan( x. Data() ) );
};

template< class T > const tnlReal< T > acos( const tnlReal< T >& x )
{
   tnl_flops_counter. recordFunction();
   return tnlReal< T >( acos( x. Data() ) );
};

template< class T > const tnlReal< T > asin( const tnlReal< T >& x )
{
   tnl_flops_counter. recordFunction();
   return tnlReal< T >( asin( x. Data() ) );
};

template< class T > const tnlReal< T > atan( const tnlReal< T >& x )
{
   tnl_flops_counter. recordFunction();
   return tnlReal< T >( atan( x. Data() ) );
};

template< class T > const tnlReal< T > atan2( const tnlReal< T >& x, const tnlReal< T >& exp )
{
   tnl_flops_counter. recordFunction();
   return tnlReal< T >( atan2( x. Data(), exp. Data() ) );
};

template< class T > const tnlReal< T > atan2( const tnlReal< T >& x, const T& exp )
{
   tnl_flops_counter. recordFunction();
   return tnlReal< T >( atan2( x. Data(), exp ) );
};


template< class T > const tnlReal< T > cosh( const tnlReal< T >& x )
{
   tnl_flops_counter. recordFunction();
   return tnlReal< T >( cosh( x. Data() ) );
};

template< class T > const tnlReal< T > sinh( const tnlReal< T >& x )
{
   tnl_flops_counter. recordFunction();
   return tnlReal< T >( sinh( x. Data() ) );
};

template< class T > const tnlReal< T > tanh( const tnlReal< T >& x )
{
   tnl_flops_counter. recordFunction();
   return tnlReal< T >( tanh( x. Data() ) );
};


template< class T > const tnlReal< T > exp( const tnlReal< T >& x )
{
   tnl_flops_counter. recordFunction();
   return tnlReal< T >( exp( x. Data() ) );
};

template< class T > const tnlReal< T > log( const tnlReal< T >& x )
{
   tnl_flops_counter. recordFunction();
   return tnlReal< T >( log( x. Data() ) );
};

template< class T > const tnlReal< T > log10( const tnlReal< T >& x )
{
   tnl_flops_counter. recordFunction();
   return tnlReal< T >( log10( x. Data() ) );
};

template< class T >
std::ostream& operator << ( std::ostream& str, const tnlReal< T >& v )
{
   str << v. Data();
   return str;
};

typedef tnlReal< float > tnlFloat;
typedef tnlReal< double > tnlDouble;

#endif /* TNLREAL_H_ */
