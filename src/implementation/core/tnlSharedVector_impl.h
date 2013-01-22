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

#include <implementation/core/vector-operations.h>

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlSharedVector< Real, Device, Index > :: getType() const
{
   return tnlString( "tnlSharedVector< " ) +
                     getParameterType< Real >() + ", " +
                     Device :: getDeviceType() + ", " +
                     getParameterType< Index >() + " >";
};

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

template< typename Element,
          typename Device,
          typename Index >
bool tnlSharedVector< Element, Device, Index > :: save( tnlFile& file ) const
{
   tnlAssert( this -> size != 0,
              cerr << "You try to save empty vector. Its name is " << this -> getName() );
   if( ! tnlObject :: save( file ) )
      return false;
   if( ! file. write( &this -> size, 1 ) )
      return false;
   if( ! file. write< Element, Device, Index >( this -> data, this -> size ) )
   {
      cerr << "I was not able to SAVE tnlSharedVector " << this -> getName()
           << " with size " << this -> getSize() << endl;
      return false;
   }
   return true;
};

template< typename Element,
          typename Device,
          typename Index >
bool tnlSharedVector< Element, Device, Index > :: save( const tnlString& fileName ) const
{
   return tnlObject :: save( fileName );
};

template< typename Real,
          typename Device,
          typename Index >
Real tnlSharedVector< Real, Device, Index > :: max() const
{
   return getVectorMax( *this );
}

template< typename Real,
          typename Device,
          typename Index >
Real tnlSharedVector< Real, Device, Index > :: min() const
{
   return getVectorMin( *this );
}


template< typename Real,
          typename Device,
          typename Index >
Real tnlSharedVector< Real, Device, Index > :: absMax() const
{
   return getVectorAbsMax( *this );
}

template< typename Real,
          typename Device,
          typename Index >
Real tnlSharedVector< Real, Device, Index > :: absMin() const
{
   return getVectorAbsMin( *this );
}

template< typename Real,
          typename Device,
          typename Index >
Real tnlSharedVector< Real, Device, Index > :: lpNorm( const Real& p ) const
{
   return getVectorLpNorm( *this, p );
}


template< typename Real,
          typename Device,
          typename Index >
Real tnlSharedVector< Real, Device, Index > :: sum() const
{
   return getVectorSum( *this );
}


template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
Real tnlSharedVector< Real, Device, Index > :: differenceMax( const Vector& v ) const
{
   return getVectorDifferenceMax( *this, v );
}


template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
Real tnlSharedVector< Real, Device, Index > :: differenceMin( const Vector& v ) const
{
   return getVectorDifferenceMin( *this, v );
}


template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
Real tnlSharedVector< Real, Device, Index > :: differenceAbsMax( const Vector& v ) const
{
   return getVectorDifferenceAbsMax( *this, v );
}

template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
Real tnlSharedVector< Real, Device, Index > :: differenceAbsMin( const Vector& v ) const
{
   return getVectorDifferenceAbsMin( *this, v );
}

template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
Real tnlSharedVector< Real, Device, Index > :: differenceLpNorm( const Vector& v, const Real& p ) const
{
   return getVectorDifferenceLpNorm( *this, v, p );
}


template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
Real tnlSharedVector< Real, Device, Index > :: differenceSum( const Vector& v ) const
{
   return getVectorDifferenceSum( *this, v );
}


template< typename Real,
          typename Device,
          typename Index >
void tnlSharedVector< Real, Device, Index > :: scalarMultiplication( const Real& alpha )
{
   vectorScalarMultiplication( *this, alpha );
}


template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
Real tnlSharedVector< Real, Device, Index > :: sdot( const Vector& v )
{
   return getVectorSdot( *this, v );
}


template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
void tnlSharedVector< Real, Device, Index > :: saxpy( const Real& alpha,
                                                      const Vector& x )
{
   vectorSaxpy( *this, x, alpha );
}

template< typename Real,
          typename Device,
          typename Index >
template< typename Vector >
void tnlSharedVector< Real, Device, Index > :: saxmy( const Real& alpha,
                                                      const Vector& x )
{
   vectorSaxmy( *this, x, alpha );
}


template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
void tnlSharedVector< Real, Device, Index > :: saxpsby( const Real& alpha,
                                                        const Vector& x,
                                                        const Real& beta )
{
      vectorSaxpsby( *this, x, alpha, beta );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
void tnlSharedVector< Real, Device, Index > :: saxpsbz( const Real& alpha,
                                                        const Vector& x,
                                                        const Real& beta,
                                                        const Vector& z )
{
      vectorSaxpsbz( *this, x, alpha, z, beta );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
void tnlSharedVector< Real, Device, Index > :: saxpsbzpy( const Real& alpha,
                                                          const Vector& x,
                                                          const Real& beta,
                                                          const Vector& z )
{
      vectorSaxpsbz( *this, x, alpha, z, beta );
}



#endif /* TNLSHAREDVECTOR_H_IMPLEMENTATION */
