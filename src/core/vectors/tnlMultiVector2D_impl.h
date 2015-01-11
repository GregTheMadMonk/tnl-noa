/***************************************************************************
                          tnlMultiVector2D_impl.h  -  description
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

#ifndef TNLMULTIVECTOR2D_IMPL_H_
#define TNLMULTIVECTOR2D_IMPL_H_



template< typename Real, typename Device, typename Index >
tnlMultiVector< 2, Real, Device, Index > :: tnlMultiVector()
{
}

template< typename Real, typename Device, typename Index >
tnlMultiVector< 2, Real, Device, Index > :: tnlMultiVector( const tnlString& name )
{
   this -> setName( name );
}


template< typename Real, typename Device, typename Index >
tnlString tnlMultiVector< 2, Real, Device, Index > :: getType()
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
tnlString tnlMultiVector< 2, Real, Device, Index > :: getTypeVirtual() const
{
   return this->getType();
};

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlMultiVector< 2, Real, Device, Index > :: getSerializationType()
{
   return HostType::getType();
};

template< typename Real,
          typename Device,
          typename Index >
tnlString tnlMultiVector< 2, Real, Device, Index > :: getSerializationTypeVirtual() const
{
   return this->getSerializationType();
};

template< typename Real, typename Device, typename Index >
bool tnlMultiVector< 2, Real, Device, Index > :: setDimensions( const Index jSize,
                                                                       const Index iSize )
{
   tnlAssert( iSize > 0 && jSize > 0,
              cerr << "iSize = " << iSize
                   << "jSize = " << jSize );

   dimensions[ 0 ] = iSize;
   dimensions[ 1 ] = jSize;
   return tnlVector< Real, Device, Index > :: setSize( iSize * jSize );
}

template< typename Real, typename Device, typename Index >
bool tnlMultiVector< 2, Real, Device, Index > :: setDimensions( const tnlStaticVector< 2, Index >& dimensions )
{
   tnlAssert( dimensions[ 0 ] > 0 && dimensions[ 1 ] > 0,
              cerr << "dimensions = " << dimensions );
   this -> dimensions = dimensions;
   return tnlVector< Real, Device, Index > :: setSize( this -> dimensions[ 1 ] * this -> dimensions[ 0 ] );
}

template< typename Real, typename Device, typename Index >
   template< typename MultiVector >
bool tnlMultiVector< 2, Real, Device, Index > :: setLike( const MultiVector& multiVector )
{
   return setDimensions( multiVector. getDimensions() );
}

template< typename Real, typename Device, typename Index >
void tnlMultiVector< 2, Real, Device, Index > :: getDimensions( Index& jSize, Index& iSize ) const
{
   iSize = this -> dimensions[ 0 ];
   jSize = this -> dimensions[ 1 ];
}

template< typename Real, typename Device, typename Index >
const tnlStaticVector< 2, Index >& tnlMultiVector< 2, Real, Device, Index > :: getDimensions() const
{
   return this -> dimensions;
}

template< typename Real, typename Device, typename Index >
Index tnlMultiVector< 2, Real, Device, Index > :: getElementIndex( const Index j, const Index i ) const
{
   tnlAssert( i >= 0 && i < this -> dimensions[ 0 ] && j >= 0 && j < this -> dimensions[ 1 ],
              cerr << "i = " << i
                   << "j = " << j
                   << "this -> dimensions[ 0 ] = " << this -> dimensions[ 0 ]
                   << "this -> dimensions[ 1 ] = " << this -> dimensions[ 1 ] );
   return j * this -> dimensions[ 0 ] + i;
}

template< typename Real, typename Device, typename Index >
Real tnlMultiVector< 2, Real, Device, Index > :: getElement( const Index j, const Index i ) const
{
   return tnlVector< Real, Device, Index > :: getElement( getElementIndex( j, i ) );
}

template< typename Real, typename Device, typename Index >
void tnlMultiVector< 2, Real, Device, Index > :: setElement( const Index j, const Index i, Real value )
{
   tnlVector< Real, Device, Index > :: setElement( getElementIndex( j, i ), value );
}


template< typename Real, typename Device, typename Index >
Real& tnlMultiVector< 2, Real, Device, Index > :: operator()( const Index j, const Index i )
{
   return tnlVector< Real, Device, Index > :: operator[]( getElementIndex( j, i ) );
}

template< typename Real, typename Device, typename Index >
const Real& tnlMultiVector< 2, Real, Device, Index > :: operator()( const Index j, const Index i ) const
{
   return tnlVector< Real, Device, Index > :: operator[]( getElementIndex( j, i ) );
}

template< typename Real, typename Device, typename Index >
   template< typename MultiVector >
bool tnlMultiVector< 2, Real, Device, Index > :: operator == ( const MultiVector& Vector ) const
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
bool tnlMultiVector< 2, Real, Device, Index > :: operator != ( const MultiVector& Vector ) const
{
   return ! ( (* this ) == Vector );
}

template< typename Real, typename Device, typename Index >
tnlMultiVector< 2, Real, Device, Index >&
   tnlMultiVector< 2, Real, Device, Index > :: operator = ( const tnlMultiVector< 2, Real, Device, Index >& Vector )
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
tnlMultiVector< 2, Real, Device, Index >&
   tnlMultiVector< 2, Real, Device, Index > :: operator = ( const MultiVector& Vector )
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
bool tnlMultiVector< 2, Real, Device, Index > :: save( tnlFile& file ) const
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
bool tnlMultiVector< 2, Real, Device, Index > :: load( tnlFile& file )
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
bool tnlMultiVector< 2, Real, Device, Index > :: save( const tnlString& fileName ) const
{
   return tnlObject :: save( fileName );
}

template< typename Real, typename Device, typename Index >
bool tnlMultiVector< 2, Real, Device, Index > :: load( const tnlString& fileName )
{
   return tnlObject :: load( fileName );
}

template< typename Real, typename Device, typename Index >
ostream& operator << ( ostream& str, const tnlMultiVector< 2, Real, Device, Index >& Vector )
{
   for( Index j = 0; j < Vector. getDimensions()[ 1 ]; j ++ )
   {
      for( Index i = 0; i < Vector. getDimensions()[ 0 ]; i ++ )
      {
         str << Vector. getElement( j, i ) << " ";
      }
      str << endl;
   }
   return str;
}

#endif /* TNLMULTIVECTOR2D_IMPL_H_ */
