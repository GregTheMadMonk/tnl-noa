/***************************************************************************
                          tnlSharedVector.h  -  description
                             -------------------
    begin                : Nov 8, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLSHAREDVECTOR_H_IMPLEMENTATION
#define TNLSHAREDVECTOR_H_IMPLEMENTATION

#include <core/vectors/tnlVectorOperations.h>

template< typename Real,
          typename Device,
          typename Index >
tnlSharedVector< Real, Device, Index >::tnlSharedVector()
{
}

template< typename Real,
          typename Device,
          typename Index >
tnlSharedVector< Real, Device, Index >::tnlSharedVector( Real* data,
                                                         const Index size )
: tnlSharedArray< Real, Device, Index >( data, size )
{
}

template< typename Real,
          typename Device,
          typename Index >
tnlSharedVector< Real, Device, Index >::tnlSharedVector( tnlVector< Real, Device, Index >& vector )
: tnlSharedArray< Real, Device, Index >( vector )
{
}

template< typename Real,
          typename Device,
          typename Index >
tnlSharedVector< Real, Device, Index >::tnlSharedVector( tnlSharedVector< Real, Device, Index >& vector )
: tnlSharedArray< Real, Device, Index >( vector )
{
}

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlSharedVector< Real, Device, Index > :: getType()
{
   return tnlString( "tnlSharedVector< " ) +
                     ::getType< Real >() + ", " +
                     Device :: getDeviceType() + ", " +
                     ::getType< Index >() + " >";
};

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlSharedVector< Real, Device, Index > :: getTypeVirtual() const
{
   return this->getType();
};

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlSharedVector< Real, Device, Index > :: getSerializationType()
{
   return tnlVector< Real, tnlHost, Index >::getType();
};

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlSharedVector< Real, Device, Index > :: getSerializationTypeVirtual() const
{
   return this->getSerializationType();
};

template< typename Real,
          typename Device,
          typename Index >
void tnlSharedVector< Real, Device, Index >::addElement( const IndexType i,
                                                         const RealType& value )
{
   tnlVectorOperations< Device >::addElement( *this, i, value );
}

template< typename Real,
          typename Device,
          typename Index >
void tnlSharedVector< Real, Device, Index >::addElement( const IndexType i,
                                                         const RealType& value,
                                                         const RealType& thisElementMultiplicator )
{
   tnlVectorOperations< Device >::addElement( *this, i, value, thisElementMultiplicator );
}

template< typename Real,
           typename Device,
           typename Index >
tnlSharedVector< Real, Device, Index >&
   tnlSharedVector< Real, Device, Index > :: operator = ( const tnlSharedVector< Real, Device, Index >& vector )
{
   tnlSharedArray< Real, Device, Index > :: operator = ( vector );
   return ( *this );
};

template< typename Real,
           typename Device,
           typename Index >
   template< typename Vector >
tnlSharedVector< Real, Device, Index >&
   tnlSharedVector< Real, Device, Index > :: operator = ( const Vector& vector )
{
   tnlSharedArray< Real, Device, Index > :: operator = ( vector );
   return ( *this );
};

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
bool tnlSharedVector< Real, Device, Index > :: operator == ( const Vector& vector ) const
{
   return tnlSharedArray< Real, Device, Index > :: operator == ( vector );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
bool tnlSharedVector< Real, Device, Index > :: operator != ( const Vector& vector ) const
{
   return tnlSharedArray< Real, Device, Index > :: operator == ( vector );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
tnlSharedVector< Real, Device, Index >& tnlSharedVector< Real, Device, Index > :: operator -= ( const Vector& vector )
{
   this->addVector( vector, -1.0 );
   return ( *this );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
tnlSharedVector< Real, Device, Index >& tnlSharedVector< Real, Device, Index > :: operator += ( const Vector& vector )
{
   this->addVector( vector );
   return ( *this );
}

template< typename Real,
          typename Device,
          typename Index >
tnlSharedVector< Real, Device, Index >& tnlSharedVector< Real, Device, Index > :: operator *= ( const RealType& c )
{
   tnlVectorOperations< Device >::vectorScalarMultiplication( *this, c );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
tnlSharedVector< Real, Device, Index >& tnlSharedVector< Real, Device, Index > :: operator /= ( const RealType& c )
{
   tnlVectorOperations< Device >::vectorScalarMultiplication( *this, 1.0/ c );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
Real tnlSharedVector< Real, Device, Index > :: max() const
{
   return tnlVectorOperations< Device > :: getVectorMax( *this );
}

template< typename Real,
          typename Device,
          typename Index >
Real tnlSharedVector< Real, Device, Index > :: min() const
{
   return tnlVectorOperations< Device > :: getVectorMin( *this );
}


template< typename Real,
          typename Device,
          typename Index >
Real tnlSharedVector< Real, Device, Index > :: absMax() const
{
   return tnlVectorOperations< Device > :: getVectorAbsMax( *this );
}

template< typename Real,
          typename Device,
          typename Index >
Real tnlSharedVector< Real, Device, Index > :: absMin() const
{
   return tnlVectorOperations< Device > :: getVectorAbsMin( *this );
}

template< typename Real,
          typename Device,
          typename Index >
Real tnlSharedVector< Real, Device, Index > :: lpNorm( const Real& p ) const
{
   return tnlVectorOperations< Device > :: getVectorLpNorm( *this, p );
}


template< typename Real,
          typename Device,
          typename Index >
Real tnlSharedVector< Real, Device, Index > :: sum() const
{
   return tnlVectorOperations< Device > :: getVectorSum( *this );
}


template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
Real tnlSharedVector< Real, Device, Index > :: differenceMax( const Vector& v ) const
{
   return tnlVectorOperations< Device > :: getVectorDifferenceMax( *this, v );
}


template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
Real tnlSharedVector< Real, Device, Index > :: differenceMin( const Vector& v ) const
{
   return tnlVectorOperations< Device > :: getVectorDifferenceMin( *this, v );
}


template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
Real tnlSharedVector< Real, Device, Index > :: differenceAbsMax( const Vector& v ) const
{
   return tnlVectorOperations< Device > :: getVectorDifferenceAbsMax( *this, v );
}

template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
Real tnlSharedVector< Real, Device, Index > :: differenceAbsMin( const Vector& v ) const
{
   return tnlVectorOperations< Device > :: getVectorDifferenceAbsMin( *this, v );
}

template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
Real tnlSharedVector< Real, Device, Index > :: differenceLpNorm( const Vector& v, const Real& p ) const
{
   return tnlVectorOperations< Device > :: getVectorDifferenceLpNorm( *this, v, p );
}


template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
Real tnlSharedVector< Real, Device, Index > :: differenceSum( const Vector& v ) const
{
   return tnlVectorOperations< Device > :: getVectorDifferenceSum( *this, v );
}


template< typename Real,
          typename Device,
          typename Index >
void tnlSharedVector< Real, Device, Index > :: scalarMultiplication( const Real& alpha )
{
   tnlVectorOperations< Device > :: vectorScalarMultiplication( *this, alpha );
}


template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
Real tnlSharedVector< Real, Device, Index > :: scalarProduct( const Vector& v )
{
   return tnlVectorOperations< Device > :: getScalarProduct( *this, v );
}

template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
void tnlSharedVector< Real, Device, Index > :: addVector( const Vector& x,
                                                          const Real& alpha,
                                                          const Real& thisMultiplicator )
{
   tnlVectorOperations< Device > :: addVector( *this, x, alpha, thisMultiplicator );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
void
tnlSharedVector< Real, Device, Index >::
addVectors( const Vector& v1,
            const Real& multiplicator1,
            const Vector& v2,
            const Real& multiplicator2,
            const Real& thisMultiplicator )
{
   tnlVectorOperations< Device >::addVectors( *this, v1, multiplicator1, v2, multiplicator2, thisMultiplicator );
}

template< typename Real,
          typename Device,
          typename Index >
void tnlSharedVector< Real, Device, Index > :: computePrefixSum()
{
   tnlVectorOperations< Device >::computePrefixSum( *this, 0, this->getSize() );
}

template< typename Real,
          typename Device,
          typename Index >
void tnlSharedVector< Real, Device, Index > :: computePrefixSum( const IndexType begin,
                                                                 const IndexType end )
{
   tnlVectorOperations< Device >::computePrefixSum( *this, begin, end );
}

template< typename Real,
          typename Device,
          typename Index >
void tnlSharedVector< Real, Device, Index > :: computeExclusivePrefixSum()
{
   tnlVectorOperations< Device >::computeExclusivePrefixSum( *this, 0, this->getSize() );
}

template< typename Real,
          typename Device,
          typename Index >
void tnlSharedVector< Real, Device, Index > :: computeExclusivePrefixSum( const IndexType begin,
                                                                          const IndexType end )
{
   tnlVectorOperations< Device >::computeExclusivePrefixSum( *this, begin, end );
}


#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

#ifdef INSTANTIATE_FLOAT
extern template class tnlSharedVector< float, tnlHost, int >;
#endif
extern template class tnlSharedVector< double, tnlHost, int >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class tnlSharedVector< long double, tnlHost, int >;
#endif
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
extern template class tnlSharedVector< float, tnlHost, long int >;
#endif
extern template class tnlSharedVector< double, tnlHost, long int >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class tnlSharedVector< long double, tnlHost, long int >;
#endif
#endif

#ifdef HAVE_CUDA
// TODO: fix this - it does not work with CUDA 5.5
/*
#ifdef INSTANTIATE_FLOAT
extern template class tnlSharedVector< float, tnlCuda, int >;
#endif
extern template class tnlSharedVector< double, tnlCuda, int >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class tnlSharedVector< long double, tnlCuda, int >;
#endif
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
extern template class tnlSharedVector< float, tnlCuda, long int >;
#endif
extern template class tnlSharedVector< double, tnlCuda, long int >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class tnlSharedVector< long double, tnlCuda, long int >;
#endif
 #endif 
 */
#endif

#endif

#endif /* TNLSHAREDVECTOR_H_IMPLEMENTATION */
