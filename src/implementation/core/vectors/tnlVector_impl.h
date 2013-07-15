/***************************************************************************
                          tnlVector.h  -  description
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

#ifndef TNLVECTOR_H_IMPLEMENTATION
#define TNLVECTOR_H_IMPLEMENTATION

#include <core/vectors/tnlVectorOperations.h>

template< typename Real,
          typename Device,
          typename Index >
tnlVector< Real, Device, Index > :: tnlVector()
{

}

template< typename Real,
          typename Device,
          typename Index >
tnlVector< Real, Device, Index > :: tnlVector( const tnlString& name )
{
   this -> setName( name );
}

template< typename Real,
          typename Device,
          typename Index >
tnlVector< Real, Device, Index > :: tnlVector( const tnlString& name, const Index size )
{
   this -> setName( name );
   this -> setSize( size );
}


template< typename Real,
          typename Device,
          typename Index >
tnlString tnlVector< Real, Device, Index > :: getType()
{
   return tnlString( "tnlVector< " ) +
                     getParameterType< Real >() + ", " +
                     Device :: getDeviceType() + ", " +
                     getParameterType< Index >() + " >";
};

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlVector< Real, Device, Index > :: getTypeVirtual() const
{
   return this->getType();
};


template< typename Real,
           typename Device,
           typename Index >
tnlVector< Real, Device, Index >&
   tnlVector< Real, Device, Index > :: operator = ( const tnlVector< Real, Device, Index >& vector )
{
   tnlArray< Real, Device, Index > :: operator = ( vector );
   return ( *this );
};

template< typename Real,
           typename Device,
           typename Index >
   template< typename Vector >
tnlVector< Real, Device, Index >&
   tnlVector< Real, Device, Index > :: operator = ( const Vector& vector )
{
   tnlArray< Real, Device, Index > :: operator = ( vector );
   return ( *this );
};

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
bool tnlVector< Real, Device, Index > :: operator == ( const Vector& vector ) const
{
   return tnlArray< Real, Device, Index > :: operator == ( vector );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
bool tnlVector< Real, Device, Index > :: operator != ( const Vector& vector ) const
{
   return tnlArray< Real, Device, Index > :: operator == ( vector );
}

template< typename Real,
          typename Device,
          typename Index >
Real tnlVector< Real, Device, Index > :: max() const
{
   return tnlVectorOperations< Device > :: getVectorMax( *this );
}

template< typename Real,
          typename Device,
          typename Index >
Real tnlVector< Real, Device, Index > :: min() const
{
   return tnlVectorOperations< Device > :: getVectorMin( *this );
}


template< typename Real,
          typename Device,
          typename Index >
Real tnlVector< Real, Device, Index > :: absMax() const
{
   return tnlVectorOperations< Device > :: getVectorAbsMax( *this );
}

template< typename Real,
          typename Device,
          typename Index >
Real tnlVector< Real, Device, Index > :: absMin() const
{
   return tnlVectorOperations< Device > :: getVectorAbsMin( *this );
}

template< typename Real,
          typename Device,
          typename Index >
Real tnlVector< Real, Device, Index > :: lpNorm( const Real& p ) const
{
   return tnlVectorOperations< Device > :: getVectorLpNorm( *this, p );
}


template< typename Real,
          typename Device,
          typename Index >
Real tnlVector< Real, Device, Index > :: sum() const
{
   return tnlVectorOperations< Device > :: getVectorSum( *this );
}


template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
Real tnlVector< Real, Device, Index > :: differenceMax( const Vector& v ) const
{
   return tnlVectorOperations< Device > :: getVectorDifferenceMax( *this, v );
}


template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
Real tnlVector< Real, Device, Index > :: differenceMin( const Vector& v ) const
{
   return tnlVectorOperations< Device > :: getVectorDifferenceMin( *this, v );
}


template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
Real tnlVector< Real, Device, Index > :: differenceAbsMax( const Vector& v ) const
{
   return tnlVectorOperations< Device > :: getVectorDifferenceAbsMax( *this, v );
}

template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
Real tnlVector< Real, Device, Index > :: differenceAbsMin( const Vector& v ) const
{
   return tnlVectorOperations< Device > :: getVectorDifferenceAbsMin( *this, v );
}

template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
Real tnlVector< Real, Device, Index > :: differenceLpNorm( const Vector& v, const Real& p ) const
{
   return tnlVectorOperations< Device > :: getVectorDifferenceLpNorm( *this, v, p );
}


template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
Real tnlVector< Real, Device, Index > :: differenceSum( const Vector& v ) const
{
   return tnlVectorOperations< Device > :: getVectorDifferenceSum( *this, v );
}


template< typename Real,
          typename Device,
          typename Index >
void tnlVector< Real, Device, Index > :: scalarMultiplication( const Real& alpha )
{
   tnlVectorOperations< Device > :: vectorScalarMultiplication( *this, alpha );
}


template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
Real tnlVector< Real, Device, Index > :: sdot( const Vector& v )
{
   return tnlVectorOperations< Device > :: getVectorSdot( *this, v );
}


template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
void tnlVector< Real, Device, Index > :: saxpy( const Real& alpha,
                                                      const Vector& x )
{
   tnlVectorOperations< Device > :: vectorSaxpy( *this, x, alpha );
}

template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
void tnlVector< Real, Device, Index > :: saxmy( const Real& alpha,
                                                const Vector& x )
{
   tnlVectorOperations< Device > :: vectorSaxmy( *this, x, alpha );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
void tnlVector< Real, Device, Index > :: saxpsby( const Real& alpha,
                                                  const Vector& x,
                                                  const Real& beta )
{
   tnlVectorOperations< Device > :: vectorSaxpsby( *this, x, alpha, beta );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
void tnlVector< Real, Device, Index > :: saxpsbz( const Real& alpha,
                                                  const Vector& x,
                                                  const Real& beta,
                                                  const Vector& z )
{
   tnlVectorOperations< Device > :: vectorSaxpsbz( *this, x, alpha, z, beta );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
void tnlVector< Real, Device, Index > :: saxpsbzpy( const Real& alpha,
                                                    const Vector& x,
                                                    const Real& beta,
                                                    const Vector& z )
{
   tnlVectorOperations< Device > :: vectorSaxpsbzpy( *this, x, alpha, z, beta );
}


#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

extern template class tnlVector< float, tnlHost, int >;
extern template class tnlVector< double, tnlHost, int >;
extern template class tnlVector< float, tnlHost, long int >;
extern template class tnlVector< double, tnlHost, long int >;

#ifdef HAVE_CUDA
/*extern template class tnlVector< float, tnlCuda, int >;
extern template class tnlVector< double, tnlCuda, int >;
extern template class tnlVector< float, tnlCuda, long int >;
extern template class tnlVector< double, tnlCuda, long int >;*/
#endif

#endif

#endif /* TNLVECTOR_H_IMPLEMENTATION */
