#pragma once

#include <TNL/Math.h>

template< typename Vector >
void setLinearSequence( Vector& deviceVector )
{
#ifdef STATIC_VECTOR
   Vector a;
#else
   typename Vector::HostType a;
   a.setLike( deviceVector );
#endif
   for( int i = 0; i < a.getSize(); i++ )
      a[ i ] = i;
   deviceVector = a;
}

template< typename Vector >
void setConstantSequence( Vector& deviceVector,
                          typename Vector::RealType v )
{
   deviceVector.setValue( v );
}

template< typename Vector >
void setOscilatingLinearSequence( Vector& deviceVector )
{
   typename Vector::HostType a;
   a.setLike( deviceVector );
   for( int i = 0; i < a.getSize(); i++ )
      a[ i ] = i % 30 - 15;
   deviceVector = a;
}

template< typename Vector >
void setOscilatingConstantSequence( Vector& deviceVector,
                                    typename Vector::RealType v )
{
   typename Vector::HostType a;
   a.setLike( deviceVector );
   for( int i = 0; i < a.getSize(); i++ )
      a[ i ] = TNL::sign( i % 30 - 15 );
   deviceVector = a;
}

template< typename Vector >
void setNegativeLinearSequence( Vector& deviceVector )
{
   typename Vector::HostType a;
   a.setLike( deviceVector );
   for( int i = 0; i < a.getSize(); i++ )
      a[ i ] = -i;
   deviceVector = a;
}

template< typename Vector >
void setOscilatingSequence( Vector& deviceVector,
                            typename Vector::RealType v )
{
#ifdef STATIC_VECTOR
   Vector a;
#else
   typename Vector::HostType a;
   a.setLike( deviceVector );
#endif
   a[ 0 ] = v;
   for( int i = 1; i < a.getSize(); i++ )
      a[ i ] = a[ i-1 ] * -1;
   deviceVector = a;
}
