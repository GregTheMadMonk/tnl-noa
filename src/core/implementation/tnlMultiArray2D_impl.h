/***************************************************************************
                          tnlMultiArray2D_impl.h  -  description
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

#ifndef TNLMULTIARRAY2D_IMPL_H_
#define TNLMULTIARRAY2D_IMPL_H_



template< typename Element, typename Device, typename Index >
tnlMultiArray< 2, Element, Device, Index > :: tnlMultiArray()
{
}

template< typename Element, typename Device, typename Index >
tnlMultiArray< 2, Element, Device, Index > :: tnlMultiArray( const tnlString& name )
{
   this -> setName( name );
}


template< typename Element, typename Device, typename Index >
tnlString tnlMultiArray< 2, Element, Device, Index > :: getType() const
{
   return tnlString( "tnlMultiArray< ") +
          tnlString( Dimensions ) +
          tnlString( ", " ) +
          tnlString( getParameterType< Element >() ) +
          tnlString( ", " ) +
          tnlString( Device :: getDeviceType() ) +
          tnlString( ", " ) +
          tnlString( getParameterType< Index >() ) +
          tnlString( " >" );
}

template< typename Element, typename Device, typename Index >
bool tnlMultiArray< 2, Element, Device, Index > :: setDimensions( const Index jSize,
                                                                       const Index iSize )
{
   tnlAssert( xSize > 0 && ySize > 0,
              cerr << "iSize = " << iSize
                   << "jSize = " << jSize );

   dimensions[ 0 ] = iSize;
   dimensions[ 1 ] = jSize;
   return tnlArray< Element, Device, Index > :: setSize( iSize * jSize );
}

template< typename Element, typename Device, typename Index >
bool tnlMultiArray< 2, Element, Device, Index > :: setDimensions( const tnlTuple< 2, Index >& dimensions )
{
   tnlAssert( dimensions[ 0 ] > 0 && dimensions[ 1 ] > 0,
              cerr << "dimensions = " << dimensions );
   this -> dimensions = dimensions;
   return tnlArray< Element, Device, Index > :: setSize( this -> dimensions[ 1 ] * this -> dimensions[ 0 ] );
}

template< typename Element, typename Device, typename Index >
   template< typename MultiArray >
bool tnlMultiArray< 2, Element, Device, Index > :: setLike( const tnlMultiArray& multiArray )
{
   return setDimensions( multiArray. getDimensions() );
}

template< typename Element, typename Device, typename Index >
void tnlMultiArray< 2, Element, Device, Index > :: getDimensions( Index& jSize, Index& iSize ) const
{
   iSize = this -> dimensions[ 0 ];
   jSize = this -> dimensions[ 1 ];
}

template< typename Element, typename Device, typename Index >
const tnlTuple< 2, Index >& tnlMultiArray< 2, Element, Device, Index > :: getDimensions() const
{
   return this -> dimensions;
}

template< typename Element, typename Device, typename Index >
Index tnlMultiArray< 2, Element, Device, Index > :: getElementIndex( const Index j, const Index i ) const
{
   tnlAssert( i >= 0 && i < this -> dimensions[ 0 ] && j >= 0 && j < this -> dimensions[ 1 ],
              cerr << "i = " << i
                   << "j = " << j
                   << "this -> dimensions[ 0 ] = " << this -> dimensions[ 0 ]
                   << "this -> dimensions[ 1 ] = " << this -> dimensions[ 1 ] );
   return j * this -> dimensions[ 0 ] + i;
}

template< typename Element, typename Device, typename Index >
Element tnlMultiArray< 2, Element, Device, Index > :: getElement( const Index j, const Index i ) const
{
   return tnlArray< Element, Device, Index > :: getElement( getElementIndex( j, i ) );
}

template< typename Element, typename Device, typename Index >
void tnlMultiArray< 2, Element, Device, Index > :: setElement( const Index j, const Index i, Element value )
{
   tnlArray< Element, Device, Index > :: setElement( getElementIndex( j, i ), value );
}


template< typename Element, typename Device, typename Index >
Element& tnlMultiArray< 2, Element, Device, Index > :: operator()( const Index j, const Index i )
{
   return tnlArray< Element, Device, Index > :: operator[]( getElementIndex( j, i ) );
}

template< typename Element, typename Device, typename Index >
const Element& tnlMultiArray< 2, Element, Device, Index > :: operator()( const Index j, const Index i ) const
{
   return tnlArray< Element, Device, Index > :: operator[]( getElementIndex( j, i ) );
}

template< typename Element, typename Device, typename Index >
   template< typename MultiArray >
bool tnlMultiArray< 2, Element, Device, Index > :: operator == ( const MultiArray& array ) const
{
   // TODO: Static assert on dimensions
   tnlAssert( this -> getDimensions() == array. getDimensions(),
              cerr << "You are attempting to compare two arrays with different dimensions." << endl
                   << "First array name is " << this -> getName()
                   << " dimensions are ( " << this -> getDimensions() << " )" << endl
                   << "Second array is " << array. getName()
                   << " dimensions are ( " << array. getDimensions() << " )" << endl; );
   return tnlArray< Element, Device, Index > :: operator == ( array );
}

template< typename Element, typename Device, typename Index >
   template< typename MultiArray >
bool tnlMultiArray< 2, Element, Device, Index > :: operator != ( const MultiArray& array ) const
{
   return ! ( (* this ) == array );
}

template< typename Element, typename Device, typename Index >
tnlMultiArray< 2, Element, Device, Index >&
   tnlMultiArray< 2, Element, Device, Index > :: operator = ( const tnlMultiArray< 2, Element, Device, Index >& array )
{
   // TODO: Static assert on dimensions
   tnlAssert( this -> getDimensions() == array. getDimensions(),
              cerr << "You are attempting to assign two arrays with different dimensions." << endl
                   << "First array name is " << this -> getName()
                   << " dimensions are ( " << this -> getDimensions() << " )" << endl
                   << "Second array is " << array. getName()
                   << " dimensions are ( " << array. getDimensions() << " )" << endl; );
   tnlArray< Element, Device, Index > :: operator = ( array );
   return ( *this );
}

template< typename Element, typename Device, typename Index >
   template< typename MultiArray >
tnlMultiArray< 2, Element, Device, Index >&
   tnlMultiArray< 2, Element, Device, Index > :: operator = ( const MultiArray& array )
{
   // TODO: Static assert on dimensions
   tnlAssert( this -> getDimensions() == array. getDimensions(),
              cerr << "You are attempting to assign two arrays with different dimensions." << endl
                   << "First array name is " << this -> getName()
                   << " dimensions are ( " << this -> getDimensions() << " )" << endl
                   << "Second array is " << array. getName()
                   << " dimensions are ( " << array. getDimensions() << " )" << endl; );
   tnlArray< Element, Device, Index > :: operator = ( array );
   return ( *this );
}

template< typename Element, typename Device, typename Index >
bool tnlMultiArray< 2, Element, Device, Index > :: save( tnlFile& file ) const
{
   if( ! tnlArray< Element, Device, Index > :: save( file ) )
   {
      cerr << "I was not able to write the tnlArray of tnlMultiArray "
           << this -> getName() << endl;
      return false;
   }
   if( ! dimensions. save( file ) )
   {
      cerr << "I was not able to write the dimensions of tnlMultiArray "
           << this -> getName() << endl;
      return false;
   }
   return true;
}

template< typename Element, typename Device, typename Index >
bool tnlMultiArray< 2, Element, Device, Index > :: load( tnlFile& file )
{
   if( ! tnlArray< Element, Device, Index > :: load( file ) )
   {
      cerr << "I was not able to read the tnlArray of tnlMultiArray "
           << this -> getName() << endl;
      return false;
   }
   if( ! dimensions. load( file ) )
   {
      cerr << "I was not able to read the dimensions of tnlMultiArray "
           << this -> getName() << endl;
      return false;
   }
   return true;
}

template< typename Element, typename Device, typename Index >
ostream& operator << ( ostream& str, const tnlMultiArray< 2, Element, Device, Index >& array )
{
   for( Index j = 0; j < array. getDimensions()[ 1 ]; j ++ )
   {
      for( Index i = 0; i < array. getDimensions()[ 0 ]; i ++ )
      {
         str << array. getElement( j, i ) << " ";
      }
      str << endl;
   }
   return str;
}

#endif /* TNLMULTIARRAY2D_IMPL_H_ */
