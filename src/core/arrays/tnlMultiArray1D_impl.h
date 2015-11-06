/***************************************************************************
                          tnlMultiArray1D_impl.h  -  description
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

#ifndef TNLMULTIARRAY1D_IMPL_H_
#define TNLMULTIARRAY1D_IMPL_H_

template< typename Element, typename Device, typename Index >
tnlMultiArray< 1, Element, Device, Index > :: tnlMultiArray()
{
}

/*template< typename Element, typename Device, typename Index >
tnlMultiArray< 1, Element, Device, Index > :: tnlMultiArray( const tnlString& name )
{
   this -> setName( name );
}*/

template< typename Element, typename Device, typename Index >
tnlString tnlMultiArray< 1, Element, Device, Index > :: getType()
{
   return tnlString( "tnlMultiArray< ") +
          tnlString( Dimensions ) +
          tnlString( ", " ) +
          tnlString( ::getType< Element >() ) +
          tnlString( ", " ) +
          tnlString( Device :: getDeviceType() ) +
          tnlString( ", " ) +
          tnlString( ::getType< Index >() ) +
          tnlString( " >" );
}

template< typename Element,
          typename Device,
          typename Index >
tnlString tnlMultiArray< 1, Element, Device, Index > :: getTypeVirtual() const
{
   return this->getType();
};

template< typename Element,
          typename Device,
          typename Index >
tnlString tnlMultiArray< 1, Element, Device, Index > :: getSerializationType()
{
   return HostType::getType();
};

template< typename Element,
          typename Device,
          typename Index >
tnlString tnlMultiArray< 1, Element, Device, Index > :: getSerializationTypeVirtual() const
{
   return this->getSerializationType();
};

template< typename Element, typename Device, typename Index >
bool tnlMultiArray< 1, Element, Device, Index > :: setDimensions( const Index iSize )
{
   tnlAssert( iSize > 0,
              cerr << "iSize = " << iSize );
   dimensions[ 0 ] = iSize;
   return tnlArray< Element, Device, Index > :: setSize( iSize );
}

template< typename Element, typename Device, typename Index >
bool tnlMultiArray< 1, Element, Device, Index > :: setDimensions( const tnlStaticVector< 1, Index >& dimensions )
{
   tnlAssert( dimensions[ 0 ] > 0,
              cerr << " dimensions[ 0 ] = " << dimensions[ 0 ] );
   this -> dimensions = dimensions;
   return tnlArray< Element, Device, Index > :: setSize( this -> dimensions[ 0 ] );
}

template< typename Element, typename Device, typename Index >
   template< typename MultiArray >
bool tnlMultiArray< 1, Element, Device, Index > :: setLike( const MultiArray& multiArray )
{
   return setDimensions( multiArray. getDimensions() );
}

template< typename Element, typename Device, typename Index >
void tnlMultiArray< 1, Element, Device, Index >::reset()
{
   this->dimensions = tnlStaticVector< 1, Index >( ( Index ) 0 );
   tnlArray< Element, Device, Index >::reset();
}

template< typename Element, typename Device, typename Index >
__cuda_callable__
void tnlMultiArray< 1, Element, Device, Index > :: getDimensions( Index& xSize ) const
{
   xSize = this -> dimensions[ 0 ];
}

template< typename Element, typename Device, typename Index >
__cuda_callable__
const tnlStaticVector< 1, Index >& tnlMultiArray< 1, Element, Device, Index > :: getDimensions() const
{
   return this -> dimensions;
}

template< typename Element, typename Device, typename Index >
__cuda_callable__
Index tnlMultiArray< 1, Element, Device, Index > :: getElementIndex( const Index i ) const
{
   tnlAssert( i >= 0 && i < this -> dimensions[ 0 ],
              cerr << "i = " << i << " this -> dimensions[ 0 ] = " <<  this -> dimensions[ 0 ] );
   return i;
}

template< typename Element, typename Device, typename Index >
Element tnlMultiArray< 1, Element, Device, Index > :: getElement( const Index i ) const
{
   return tnlArray< Element, Device, Index > :: getElement( getElementIndex( i ) );
}

template< typename Element, typename Device, typename Index >
void tnlMultiArray< 1, Element, Device, Index > :: setElement( const Index i, Element value )
{
   tnlArray< Element, Device, Index > :: setElement( getElementIndex( i ), value );
}

template< typename Element, typename Device, typename Index >
__cuda_callable__
Element& tnlMultiArray< 1, Element, Device, Index > :: operator()( const Index element )
{
   return tnlArray< Element, Device, Index > :: operator[]( getElementIndex( element ) );
}

template< typename Element, typename Device, typename Index >
__cuda_callable__
const Element& tnlMultiArray< 1, Element, Device, Index > :: operator()( const Index element ) const
{
   return tnlArray< Element, Device, Index > :: operator[]( getElementIndex( element ) );
}

template< typename Element, typename Device, typename Index >
   template< typename MultiArray >
bool tnlMultiArray< 1, Element, Device, Index > :: operator == ( const MultiArray& array ) const
{
   // TODO: Static assert on dimensions
   tnlAssert( this -> getDimensions() == array. getDimensions(),
              cerr << "You are attempting to compare two arrays with different dimensions." << endl
                   << "First array dimensions are ( " << this -> getDimensions() << " )" << endl
                   << "Second array dimensions are ( " << array. getDimensions() << " )" << endl; );
   return tnlArray< Element, Device, Index > :: operator == ( array );
}

template< typename Element, typename Device, typename Index >
   template< typename MultiArray >
bool tnlMultiArray< 1, Element, Device, Index > :: operator != ( const MultiArray& array ) const
{
   return ! ( (* this ) == array );
}

template< typename Element, typename Device, typename Index >
tnlMultiArray< 1, Element, Device, Index >&
   tnlMultiArray< 1, Element, Device, Index > :: operator = ( const tnlMultiArray< 1, Element, Device, Index >& array )
{
   // TODO: Static assert on dimensions
   tnlAssert( this -> getDimensions() == array. getDimensions(),
              cerr << "You are attempting to assign two arrays with different dimensions." << endl
                   << "First array dimensions are ( " << this -> getDimensions() << " )" << endl
                   << "Second array dimensions are ( " << array. getDimensions() << " )" << endl; );
   tnlArray< Element, Device, Index > :: operator = ( array );
   return ( *this );
}

template< typename Element, typename Device, typename Index >
   template< typename MultiArray >
tnlMultiArray< 1, Element, Device, Index >&
   tnlMultiArray< 1, Element, Device, Index > :: operator = ( const MultiArray& array )
{
   // TODO: Static assert on dimensions
   tnlAssert( this -> getDimensions() == array. getDimensions(),
              cerr << "You are attempting to assign two arrays with different dimensions." << endl
                   << "First array dimensions are ( " << this -> getDimensions() << " )" << endl
                   << "Second array dimensions are ( " << array. getDimensions() << " )" << endl; );
   tnlArray< Element, Device, Index > :: operator = ( array );
   return ( *this );
}

template< typename Element, typename Device, typename Index >
bool tnlMultiArray< 1, Element, Device, Index > :: save( tnlFile& file ) const
{
   if( ! tnlArray< Element, Device, Index > :: save( file ) )
   {
      cerr << "I was not able to write the tnlArray of tnlMultiArray." << endl;
      return false;
   }
   if( ! dimensions. save( file ) )
   {
      cerr << "I was not able to write the dimensions of tnlMultiArray." << endl;
      return false;
   }
   return true;
}

template< typename Element, typename Device, typename Index >
bool tnlMultiArray< 1, Element, Device, Index > :: load( tnlFile& file )
{
   if( ! tnlArray< Element, Device, Index > :: load( file ) )
   {
      cerr << "I was not able to read the tnlArray of tnlMultiArray." << endl;
      return false;
   }
   if( ! dimensions. load( file ) )
   {
      cerr << "I was not able to read the dimensions of tnlMultiArray." << endl;
      return false;
   }
   return true;
}

template< typename Element, typename Device, typename Index >
bool tnlMultiArray< 1, Element, Device, Index > :: save( const tnlString& fileName ) const
{
   return tnlObject :: save( fileName );
}

template< typename Element, typename Device, typename Index >
bool tnlMultiArray< 1, Element, Device, Index > :: load( const tnlString& fileName )
{
   return tnlObject :: load( fileName );
}

template< typename Element, typename Device, typename Index >
ostream& operator << ( ostream& str, const tnlMultiArray< 1, Element, Device, Index >& array )
{
   for( Index i = 0; i < array. getDimensions()[ 0 ]; i ++ )
   {
      str << array. getElement( i ) << " ";
   }
   return str;
}

#endif /* TNLMULTIARRAY1D_IMPL_H_ */
