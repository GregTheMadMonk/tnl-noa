/***************************************************************************
                          tnlVector.h  -  description
                             -------------------
    begin                : Nov 8, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

#include <core/vectors/tnlVectorOperations.h>

namespace TNL {

template< typename Real,
          typename Device,
          typename Index >
tnlVector< Real, Device, Index >::tnlVector()
{

}

template< typename Real,
          typename Device,
          typename Index >
tnlVector< Real, Device, Index >::tnlVector( const Index size )
{
   this->setSize( size );
}


template< typename Real,
          typename Device,
          typename Index >
tnlString tnlVector< Real, Device, Index >::getType()
{
   return tnlString( "tnlVector< " ) +
                     ::getType< Real >() + ", " +
                     Device::getDeviceType() + ", " +
                     ::getType< Index >() + " >";
};

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlVector< Real, Device, Index >::getTypeVirtual() const
{
   return this->getType();
};

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlVector< Real, Device, Index >::getSerializationType()
{
   return HostType::getType();
};

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlVector< Real, Device, Index >::getSerializationTypeVirtual() const
{
   return this->getSerializationType();
};

template< typename Real,
          typename Device,
          typename Index >
void tnlVector< Real, Device, Index >::addElement( const IndexType i,
                                                   const RealType& value )
{
   tnlVectorOperations< Device >::addElement( *this, i, value );
}

template< typename Real,
          typename Device,
          typename Index >
void tnlVector< Real, Device, Index >::addElement( const IndexType i,
                                                   const RealType& value,
                                                   const RealType& thisElementMultiplicator )
{
   tnlVectorOperations< Device >::addElement( *this, i, value, thisElementMultiplicator );
}

template< typename Real,
           typename Device,
           typename Index >
tnlVector< Real, Device, Index >&
   tnlVector< Real, Device, Index >::operator = ( const tnlVector< Real, Device, Index >& vector )
{
   tnlArray< Real, Device, Index >::operator = ( vector );
   return ( *this );
};

template< typename Real,
           typename Device,
           typename Index >
   template< typename Vector >
tnlVector< Real, Device, Index >&
   tnlVector< Real, Device, Index >::operator = ( const Vector& vector )
{
   tnlArray< Real, Device, Index >::operator = ( vector );
   return ( *this );
};

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
bool tnlVector< Real, Device, Index >::operator == ( const Vector& vector ) const
{
   return tnlArray< Real, Device, Index >::operator == ( vector );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
bool tnlVector< Real, Device, Index >::operator != ( const Vector& vector ) const
{
   return tnlArray< Real, Device, Index >::operator != ( vector );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
tnlVector< Real, Device, Index >& tnlVector< Real, Device, Index >::operator -= ( const Vector& vector )
{
   this->addVector( vector, -1.0 );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
tnlVector< Real, Device, Index >& tnlVector< Real, Device, Index >::operator += ( const Vector& vector )
{
   this->addVector( vector );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
tnlVector< Real, Device, Index >& tnlVector< Real, Device, Index >::operator *= ( const RealType& c )
{
   tnlVectorOperations< Device >::vectorScalarMultiplication( *this, c );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
tnlVector< Real, Device, Index >& tnlVector< Real, Device, Index >::operator /= ( const RealType& c )
{
   tnlVectorOperations< Device >::vectorScalarMultiplication( *this, 1.0 / c );
   return *this;
}


template< typename Real,
          typename Device,
          typename Index >
Real tnlVector< Real, Device, Index >::max() const
{
   return tnlVectorOperations< Device >::getVectorMax( *this );
}

template< typename Real,
          typename Device,
          typename Index >
Real tnlVector< Real, Device, Index >::min() const
{
   return tnlVectorOperations< Device >::getVectorMin( *this );
}


template< typename Real,
          typename Device,
          typename Index >
Real tnlVector< Real, Device, Index >::absMax() const
{
   return tnlVectorOperations< Device >::getVectorAbsMax( *this );
}

template< typename Real,
          typename Device,
          typename Index >
Real tnlVector< Real, Device, Index >::absMin() const
{
   return tnlVectorOperations< Device >::getVectorAbsMin( *this );
}

template< typename Real,
          typename Device,
          typename Index >
Real tnlVector< Real, Device, Index >::lpNorm( const Real& p ) const
{
   return tnlVectorOperations< Device >::getVectorLpNorm( *this, p );
}


template< typename Real,
          typename Device,
          typename Index >
Real tnlVector< Real, Device, Index >::sum() const
{
   return tnlVectorOperations< Device >::getVectorSum( *this );
}


template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
Real tnlVector< Real, Device, Index >::differenceMax( const Vector& v ) const
{
   return tnlVectorOperations< Device >::getVectorDifferenceMax( *this, v );
}


template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
Real tnlVector< Real, Device, Index >::differenceMin( const Vector& v ) const
{
   return tnlVectorOperations< Device >::getVectorDifferenceMin( *this, v );
}


template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
Real tnlVector< Real, Device, Index >::differenceAbsMax( const Vector& v ) const
{
   return tnlVectorOperations< Device >::getVectorDifferenceAbsMax( *this, v );
}

template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
Real tnlVector< Real, Device, Index >::differenceAbsMin( const Vector& v ) const
{
   return tnlVectorOperations< Device >::getVectorDifferenceAbsMin( *this, v );
}

template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
Real tnlVector< Real, Device, Index >::differenceLpNorm( const Vector& v, const Real& p ) const
{
   return tnlVectorOperations< Device >::getVectorDifferenceLpNorm( *this, v, p );
}


template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
Real tnlVector< Real, Device, Index >::differenceSum( const Vector& v ) const
{
   return tnlVectorOperations< Device >::getVectorDifferenceSum( *this, v );
}


template< typename Real,
          typename Device,
          typename Index >
void tnlVector< Real, Device, Index >::scalarMultiplication( const Real& alpha )
{
   tnlVectorOperations< Device >::vectorScalarMultiplication( *this, alpha );
}


template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
Real tnlVector< Real, Device, Index >::scalarProduct( const Vector& v ) const
{
   return tnlVectorOperations< Device >::getScalarProduct( *this, v );
}

template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
void tnlVector< Real, Device, Index >::addVector( const Vector& x,
                                                    const Real& multiplicator,
                                                    const Real& thisMultiplicator )
{
   tnlVectorOperations< Device >::addVector( *this, x, multiplicator, thisMultiplicator );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
void
tnlVector< Real, Device, Index >::
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
void tnlVector< Real, Device, Index >::computePrefixSum()
{
   tnlVectorOperations< Device >::computePrefixSum( *this, 0, this->getSize() );
}

template< typename Real,
          typename Device,
          typename Index >
void tnlVector< Real, Device, Index >::computePrefixSum( const IndexType begin,
                                                           const IndexType end )
{
   tnlVectorOperations< Device >::computePrefixSum( *this, begin, end );
}

template< typename Real,
          typename Device,
          typename Index >
void tnlVector< Real, Device, Index >::computeExclusivePrefixSum()
{
   tnlVectorOperations< Device >::computeExclusivePrefixSum( *this, 0, this->getSize() );
}

template< typename Real,
          typename Device,
          typename Index >
void tnlVector< Real, Device, Index >::computeExclusivePrefixSum( const IndexType begin,
                                                                    const IndexType end )
{
   tnlVectorOperations< Device >::computeExclusivePrefixSum( *this, begin, end );
}


#ifdef UNDEF //TEMPLATE_EXPLICIT_INSTANTIATION

#ifdef INSTANTIATE_FLOAT
extern template class tnlVector< float, tnlHost, int >;
extern template tnlVector< float, tnlHost, int >& tnlVector< float, tnlHost, int >:: operator = ( const tnlVector< double, tnlHost, int >& vector );
#endif

extern template class tnlVector< double, tnlHost, int >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class tnlVector< long double, tnlHost, int >;
#endif

#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
extern template class tnlVector< float, tnlHost, long int >;
#endif
extern template class tnlVector< double, tnlHost, long int >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class tnlVector< long double, tnlHost, long int >;
#endif
#endif

#ifdef HAVE_CUDA
#ifdef INSTANTIATE_FLOAT
extern template class tnlVector< float, tnlCuda, int >;
#endif
extern template class tnlVector< double, tnlCuda, int >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class tnlVector< long double, tnlCuda, int >;
#endif

#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
extern template class tnlVector< float, tnlCuda, long int >;
#endif
extern template class tnlVector< double, tnlCuda, long int >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class tnlVector< long double, tnlCuda, long int >;
#endif
#endif
#endif

#endif

} // namespace TNL
