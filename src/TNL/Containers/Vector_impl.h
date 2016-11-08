/***************************************************************************
                          Vector.h  -  description
                             -------------------
    begin                : Nov 8, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

#include <TNL/Containers/VectorOperations.h>

namespace TNL {
namespace Containers {   

template< typename Real,
          typename Device,
          typename Index >
Vector< Real, Device, Index >::Vector()
{

}

template< typename Real,
          typename Device,
          typename Index >
Vector< Real, Device, Index >::Vector( const Index size )
{
   this->setSize( size );
}


template< typename Real,
          typename Device,
          typename Index >
String Vector< Real, Device, Index >::getType()
{
   return String( "Containers::Vector< " ) +
                    TNL::getType< Real >() + ", " +
                     Device::getDeviceType() + ", " +
                    TNL::getType< Index >() + " >";
};

template< typename Real,
          typename Device,
          typename Index >
String Vector< Real, Device, Index >::getTypeVirtual() const
{
   return this->getType();
};

template< typename Real,
          typename Device,
          typename Index >
String Vector< Real, Device, Index >::getSerializationType()
{
   return HostType::getType();
};

template< typename Real,
          typename Device,
          typename Index >
String Vector< Real, Device, Index >::getSerializationTypeVirtual() const
{
   return this->getSerializationType();
};

template< typename Real,
          typename Device,
          typename Index >
void Vector< Real, Device, Index >::addElement( const IndexType i,
                                                   const RealType& value )
{
   VectorOperations< Device >::addElement( *this, i, value );
}

template< typename Real,
          typename Device,
          typename Index >
void Vector< Real, Device, Index >::addElement( const IndexType i,
                                                   const RealType& value,
                                                   const RealType& thisElementMultiplicator )
{
   VectorOperations< Device >::addElement( *this, i, value, thisElementMultiplicator );
}

template< typename Real,
           typename Device,
           typename Index >
Vector< Real, Device, Index >&
   Vector< Real, Device, Index >::operator = ( const Vector< Real, Device, Index >& vector )
{
   Containers::Array< Real, Device, Index >::operator = ( vector );
   return ( *this );
};

template< typename Real,
           typename Device,
           typename Index >
   template< typename VectorT >
Vector< Real, Device, Index >&
   Vector< Real, Device, Index >::operator = ( const VectorT& vector )
{
   Containers::Array< Real, Device, Index >::operator = ( vector );
   return ( *this );
};

template< typename Real,
          typename Device,
          typename Index >
   template< typename VectorT >
bool Vector< Real, Device, Index >::operator == ( const VectorT& vector ) const
{
   return Containers::Array< Real, Device, Index >::operator == ( vector );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename VectorT >
bool Vector< Real, Device, Index >::operator != ( const VectorT& vector ) const
{
   return Containers::Array< Real, Device, Index >::operator != ( vector );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename VectorT >
Vector< Real, Device, Index >& Vector< Real, Device, Index >::operator -= ( const VectorT& vector )
{
   this->addVector( vector, -1.0 );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename VectorT >
Vector< Real, Device, Index >& Vector< Real, Device, Index >::operator += ( const VectorT& vector )
{
   this->addVector( vector );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
Vector< Real, Device, Index >& Vector< Real, Device, Index >::operator *= ( const RealType& c )
{
   VectorOperations< Device >::vectorScalarMultiplication( *this, c );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index >
Vector< Real, Device, Index >& Vector< Real, Device, Index >::operator /= ( const RealType& c )
{
   VectorOperations< Device >::vectorScalarMultiplication( *this, 1.0 / c );
   return *this;
}


template< typename Real,
          typename Device,
          typename Index >
Real Vector< Real, Device, Index >::max() const
{
   return VectorOperations< Device >::getVectorMax( *this );
}

template< typename Real,
          typename Device,
          typename Index >
Real Vector< Real, Device, Index >::min() const
{
   return VectorOperations< Device >::getVectorMin( *this );
}


template< typename Real,
          typename Device,
          typename Index >
Real Vector< Real, Device, Index >::absMax() const
{
   return VectorOperations< Device >::getVectorAbsMax( *this );
}

template< typename Real,
          typename Device,
          typename Index >
Real Vector< Real, Device, Index >::absMin() const
{
   return VectorOperations< Device >::getVectorAbsMin( *this );
}

template< typename Real,
          typename Device,
          typename Index >
Real Vector< Real, Device, Index >::lpNorm( const Real& p ) const
{
   return VectorOperations< Device >::getVectorLpNorm( *this, p );
}


template< typename Real,
          typename Device,
          typename Index >
Real Vector< Real, Device, Index >::sum() const
{
   return VectorOperations< Device >::getVectorSum( *this );
}


template< typename Real,
          typename Device,
          typename Index >
template< typename VectorT >
Real Vector< Real, Device, Index >::differenceMax( const VectorT& v ) const
{
   return VectorOperations< Device >::getVectorDifferenceMax( *this, v );
}


template< typename Real,
          typename Device,
          typename Index >
template< typename VectorT >
Real Vector< Real, Device, Index >::differenceMin( const VectorT& v ) const
{
   return VectorOperations< Device >::getVectorDifferenceMin( *this, v );
}


template< typename Real,
          typename Device,
          typename Index >
template< typename VectorT >
Real Vector< Real, Device, Index >::differenceAbsMax( const VectorT& v ) const
{
   return VectorOperations< Device >::getVectorDifferenceAbsMax( *this, v );
}

template< typename Real,
          typename Device,
          typename Index >
template< typename VectorT >
Real Vector< Real, Device, Index >::differenceAbsMin( const VectorT& v ) const
{
   return VectorOperations< Device >::getVectorDifferenceAbsMin( *this, v );
}

template< typename Real,
          typename Device,
          typename Index >
template< typename VectorT >
Real Vector< Real, Device, Index >::differenceLpNorm( const VectorT& v, const Real& p ) const
{
   return VectorOperations< Device >::getVectorDifferenceLpNorm( *this, v, p );
}


template< typename Real,
          typename Device,
          typename Index >
template< typename VectorT >
Real Vector< Real, Device, Index >::differenceSum( const VectorT& v ) const
{
   return VectorOperations< Device >::getVectorDifferenceSum( *this, v );
}


template< typename Real,
          typename Device,
          typename Index >
void Vector< Real, Device, Index >::scalarMultiplication( const Real& alpha )
{
   VectorOperations< Device >::vectorScalarMultiplication( *this, alpha );
}


template< typename Real,
          typename Device,
          typename Index >
template< typename VectorT >
Real Vector< Real, Device, Index >::scalarProduct( const VectorT& v ) const
{
   return VectorOperations< Device >::getScalarProduct( *this, v );
}

template< typename Real,
          typename Device,
          typename Index >
template< typename VectorT >
void Vector< Real, Device, Index >::addVector( const VectorT& x,
                                                    const Real& multiplicator,
                                                    const Real& thisMultiplicator )
{
   VectorOperations< Device >::addVector( *this, x, multiplicator, thisMultiplicator );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename VectorT >
void
Vector< Real, Device, Index >::
addVectors( const VectorT& v1,
            const Real& multiplicator1,
            const VectorT& v2,
            const Real& multiplicator2,
            const Real& thisMultiplicator )
{
   VectorOperations< Device >::addVectors( *this, v1, multiplicator1, v2, multiplicator2, thisMultiplicator );
}

template< typename Real,
          typename Device,
          typename Index >
void Vector< Real, Device, Index >::computePrefixSum()
{
   VectorOperations< Device >::computePrefixSum( *this, 0, this->getSize() );
}

template< typename Real,
          typename Device,
          typename Index >
void Vector< Real, Device, Index >::computePrefixSum( const IndexType begin,
                                                           const IndexType end )
{
   VectorOperations< Device >::computePrefixSum( *this, begin, end );
}

template< typename Real,
          typename Device,
          typename Index >
void Vector< Real, Device, Index >::computeExclusivePrefixSum()
{
   VectorOperations< Device >::computeExclusivePrefixSum( *this, 0, this->getSize() );
}

template< typename Real,
          typename Device,
          typename Index >
void Vector< Real, Device, Index >::computeExclusivePrefixSum( const IndexType begin,
                                                                    const IndexType end )
{
   VectorOperations< Device >::computeExclusivePrefixSum( *this, begin, end );
}


#ifdef UNDEF //TEMPLATE_EXPLICIT_INSTANTIATION

#ifdef INSTANTIATE_FLOAT
extern template class Vector< float, Devices::Host, int >;
extern template Vector< float, Devices::Host, int >& Vector< float, Devices::Host, int >:: operator = ( const Vector< double, Devices::Host, int >& vector );
#endif

extern template class Vector< double, Devices::Host, int >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class Vector< long double, Devices::Host, int >;
#endif

#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
extern template class Vector< float, Devices::Host, long int >;
#endif
extern template class Vector< double, Devices::Host, long int >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class Vector< long double, Devices::Host, long int >;
#endif
#endif

#ifdef HAVE_CUDA
#ifdef INSTANTIATE_FLOAT
extern template class Vector< float, Devices::Cuda, int >;
#endif
extern template class Vector< double, Devices::Cuda, int >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class Vector< long double, Devices::Cuda, int >;
#endif

#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
extern template class Vector< float, Devices::Cuda, long int >;
#endif
extern template class Vector< double, Devices::Cuda, long int >;
#ifdef INSTANTIATE_LONG_DOUBLE
extern template class Vector< long double, Devices::Cuda, long int >;
#endif
#endif
#endif

#endif

} // namespace Containers
} // namespace TNL
