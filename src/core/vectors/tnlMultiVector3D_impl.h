/***************************************************************************
                          tnlMultiVector3D_impl.h  -  description
                             -------------------
    begin                : Nov 13, 2012
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

#ifndef TNLMULTIVECTOR3D_IMPL_H_
#define TNLMULTIVECTOR3D_IMPL_H_



template< typename Real, typename Device, typename Index >
tnlMultiVector< 3, Real, Device, Index > :: tnlMultiVector()
{
}

/*template< typename Real, typename Device, typename Index >
tnlMultiVector< 3, Real, Device, Index > :: tnlMultiVector( const tnlString& name )
{
   this -> setName( name );
}*/

template< typename Real, typename Device, typename Index >
tnlString tnlMultiVector< 3, Real, Device, Index > :: getType()
{
   return tnlString( "tnlMultiVector< ") +
          tnlString( Dimensions ) +
          tnlString( ", " ) +
          tnlString( ::getType< Real >() ) +
          tnlString( ", " ) +
          tnlString( Device :: getDeviceType() ) +
          tnlString( ", " ) +
          tnlString( ::getType< Index >() ) +
          tnlString( " >" );
}

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlMultiVector< 3, Real, Device, Index > :: getTypeVirtual() const
{
   return this->getType();
};

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlMultiVector< 3, Real, Device, Index > :: getSerializationType()
{
   return HostType::getType();
};

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlMultiVector< 3, Real, Device, Index > :: getSerializationTypeVirtual() const
{
   return this->getSerializationType();
};

template< typename Real, typename Device, typename Index >
bool tnlMultiVector< 3, Real, Device, Index > :: setDimensions( const Index kSize,
                                                                       const Index jSize,
                                                                       const Index iSize )
{
   tnlAssert( iSize > 0 && jSize > 0 && kSize > 0,
              cerr << "iSize = " << iSize
                   << "jSize = " << jSize
                   << "kSize = " << kSize );

   dimensions[ 0 ] = iSize;
   dimensions[ 1 ] = jSize;
   dimensions[ 2 ] = kSize;
   return tnlVector< Real, Device, Index > :: setSize( iSize * jSize * kSize );
}

template< typename Real, typename Device, typename Index >
bool tnlMultiVector< 3, Real, Device, Index > :: setDimensions( const tnlStaticVector< 3, Index >& dimensions )
{
   tnlAssert( dimensions[ 0 ] > 0 && dimensions[ 1 ] > 0 && dimensions[ 2 ],
              cerr << "dimensions = " << dimensions );
   this -> dimensions = dimensions;
   return tnlVector< Real, Device, Index > :: setSize( this -> dimensions[ 2 ] *
                                                          this -> dimensions[ 1 ] *
                                                          this -> dimensions[ 0 ] );
}

template< typename Real, typename Device, typename Index >
   template< typename MultiVector >
bool tnlMultiVector< 3, Real, Device, Index > :: setLike( const MultiVector& multiVector )
{
   return setDimensions( multiVector. getDimensions() );
}

template< typename Real, typename Device, typename Index >
void tnlMultiVector< 3, Real, Device, Index > :: getDimensions( Index& kSize,
                                                                  Index& jSize,
                                                                  Index& iSize ) const
{
   iSize = this -> dimensions[ 0 ];
   jSize = this -> dimensions[ 1 ];
   kSize = this -> dimensions[ 2 ];
}

template< typename Real, typename Device, typename Index >
const tnlStaticVector< 3, Index >& tnlMultiVector< 3, Real, Device, Index > :: getDimensions() const
{
   return this -> dimensions;
}

template< typename Real, typename Device, typename Index >
Index tnlMultiVector< 3, Real, Device, Index > :: getElementIndex( const Index k,
                                                                     const Index j,
                                                                     const Index i ) const
{
   tnlAssert( i >= 0 && i < this -> dimensions[ 0 ] &&
              j >= 0 && j < this -> dimensions[ 1 ] &&
              k >= 0 && k < this -> dimensions[ 2 ],
              cerr << " i = " << i
                   << " j = " << j
                   << " k = " << k
                   << " this -> dimensions = " << this -> dimensions );
   return ( k * this -> dimensions[ 1 ]  + j ) * this -> dimensions[ 0 ] + i;
}

template< typename Real, typename Device, typename Index >
Real tnlMultiVector< 3, Real, Device, Index > :: getElement( const Index k,
                                                                  const Index j,
                                                                  const Index i ) const
{
   return tnlVector< Real, Device, Index > :: getElement( getElementIndex( k, j, i ) );
}

template< typename Real, typename Device, typename Index >
void tnlMultiVector< 3, Real, Device, Index > :: setElement( const Index k,
                                                                    const Index j,
                                                                    const Index i, Real value )
{
   tnlVector< Real, Device, Index > :: setElement( getElementIndex( k, j, i ), value );
}


template< typename Real, typename Device, typename Index >
Real& tnlMultiVector< 3, Real, Device, Index > :: operator()( const Index k,
                                                                        const Index j,
                                                                        const Index i )
{
   return tnlVector< Real, Device, Index > :: operator[]( getElementIndex( k, j, i ) );
}

template< typename Real, typename Device, typename Index >
const Real& tnlMultiVector< 3, Real, Device, Index > :: operator()( const Index k,
                                                                               const Index j,
                                                                               const Index i ) const
{
   return tnlVector< Real, Device, Index > :: operator[]( getElementIndex( k, j, i ) );
}

template< typename Real, typename Device, typename Index >
   template< typename MultiVector >
bool tnlMultiVector< 3, Real, Device, Index > :: operator == ( const MultiVector& Vector ) const
{
   // TODO: Static assert on dimensions
   tnlAssert( this -> getDimensions() == Vector. getDimensions(),
              cerr << "You are attempting to compare two Vectors with different dimensions." << endl
                   << "First Vector name is " << this -> getName()
                   << " dimensions are ( " << this -> getDimensions() << " )" << endl
                   << "Second Vector is " << Vector. getName()
                   << " dimensions are ( " << Vector. getDimensions() << " )" << endl; );
   return tnlVector< Real, Device, Index > :: operator == ( Vector );
}

template< typename Real, typename Device, typename Index >
   template< typename MultiVector >
bool tnlMultiVector< 3, Real, Device, Index > :: operator != ( const MultiVector& Vector ) const
{
   return ! ( (* this ) == Vector );
}

template< typename Real, typename Device, typename Index >
tnlMultiVector< 3, Real, Device, Index >&
   tnlMultiVector< 3, Real, Device, Index > :: operator = ( const tnlMultiVector< 3, Real, Device, Index >& Vector )
{
   // TODO: Static assert on dimensions
   tnlAssert( this -> getDimensions() == Vector. getDimensions(),
              cerr << "You are attempting to assign two Vectors with different dimensions." << endl
                   << "First Vector name is " << this -> getName()
                   << " dimensions are ( " << this -> getDimensions() << " )" << endl
                   << "Second Vector is " << Vector. getName()
                   << " dimensions are ( " << Vector. getDimensions() << " )" << endl; );
   tnlVector< Real, Device, Index > :: operator = ( Vector );
   return ( *this );
}

template< typename Real, typename Device, typename Index >
   template< typename MultiVector >
tnlMultiVector< 3, Real, Device, Index >&
   tnlMultiVector< 3, Real, Device, Index > :: operator = ( const MultiVector& Vector )
{
   // TODO: Static assert on dimensions
   tnlAssert( this -> getDimensions() == Vector. getDimensions(),
              cerr << "You are attempting to assign two Vectors with different dimensions." << endl
                   << "First Vector name is " << this -> getName()
                   << " dimensions are ( " << this -> getDimensions() << " )" << endl
                   << "Second Vector is " << Vector. getName()
                   << " dimensions are ( " << Vector. getDimensions() << " )" << endl; );
   tnlVector< Real, Device, Index > :: operator = ( Vector );
   return ( *this );
}

template< typename Real, typename Device, typename Index >
bool tnlMultiVector< 3, Real, Device, Index > :: save( tnlFile& file ) const
{
   if( ! tnlVector< Real, Device, Index > :: save( file ) )
   {
      cerr << "I was not able to write the tnlVector of tnlMultiVector "
           << this -> getName() << endl;
      return false;
   }
   if( ! dimensions. save( file ) )
   {
      cerr << "I was not able to write the dimensions of tnlMultiVector "
           << this -> getName() << endl;
      return false;
   }
   return true;
}

template< typename Real, typename Device, typename Index >
bool tnlMultiVector< 3, Real, Device, Index > :: load( tnlFile& file )
{
   if( ! tnlVector< Real, Device, Index > :: load( file ) )
   {
      cerr << "I was not able to read the tnlVector of tnlMultiVector "
           << this -> getName() << endl;
      return false;
   }
   if( ! dimensions. load( file ) )
   {
      cerr << "I was not able to read the dimensions of tnlMultiVector "
           << this -> getName() << endl;
      return false;
   }
   return true;
}

template< typename Real, typename Device, typename Index >
ostream& operator << ( ostream& str, const tnlMultiVector< 3, Real, Device, Index >& Vector )
{
   for( Index k = 0; k < Vector. getDimensions()[ 2 ]; k ++ )
   {
      for( Index j = 0; j < Vector. getDimensions()[ 1 ]; j ++ )
      {
         for( Index i = 0; i < Vector. getDimensions()[ 0 ]; i ++ )
         {
            str << Vector. getElement( k, j, i ) << " ";
         }
         str << endl;
      }
      str << endl;
   }
   return str;
}

template< typename Real, typename Device, typename Index >
bool tnlMultiVector< 3, Real, Device, Index > :: save( const tnlString& fileName ) const
{
   return tnlObject :: save( fileName );
}

template< typename Real, typename Device, typename Index >
bool tnlMultiVector< 3, Real, Device, Index > :: load( const tnlString& fileName )
{
   return tnlObject :: load( fileName );
}

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

#ifdef INSTANTIATE_FLOAT
extern template class tnlMultiVector< 3, float,  tnlHost, int >;
#endif
extern template class tnlMultiVector< 3, double, tnlHost, int >;
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
extern template class tnlMultiVector< 3, float,  tnlHost, long int >;
#endif
extern template class tnlMultiVector< 3, double, tnlHost, long int >;
#endif

#ifdef HAVE_CUDA
/*#ifdef INSTANTIATE_FLOAT
extern template class tnlMultiVector< 3, float,  tnlCuda, int >;
#endif
extern template class tnlMultiVector< 3, double, tnlCuda, int >;
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
extern template class tnlMultiVector< 3, float,  tnlCuda, long int >;
#endif
extern template class tnlMultiVector< 3, double, tnlCuda, long int >;
#endif*/
#endif

#endif


#endif /* TNLMULTIVECTOR3D_IMPL_H_ */
