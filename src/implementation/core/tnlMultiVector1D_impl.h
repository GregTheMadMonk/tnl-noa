/***************************************************************************
                          tnlMultiVector1D_impl.h  -  description
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

#ifndef TNLMULTIVECTOT1D_IMPL_H_
#define TNLMULTIVECTOR1D_IMPL_H_



template< typename Element, typename Device, typename Index >
tnlMultiVector< 1, Element, Device, Index > :: tnlMultiVector()
{
}

template< typename Element, typename Device, typename Index >
tnlMultiVector< 1, Element, Device, Index > :: tnlMultiVector( const tnlString& name )
{
   this -> setName( name );
}

template< typename Element, typename Device, typename Index >
tnlString tnlMultiVector< 1, Element, Device, Index > :: getType() const
{
   return tnlString( "tnlMultiVector< ") +
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
bool tnlMultiVector< 1, Element, Device, Index > :: setDimensions( const Index iSize )
{
   tnlAssert( iSize > 0,
              cerr << "iSize = " << iSize );
   dimensions[ 0 ] = iSize;
   return tnlVector< Element, Device, Index > :: setSize( iSize );
}

template< typename Element, typename Device, typename Index >
bool tnlMultiVector< 1, Element, Device, Index > :: setDimensions( const tnlTuple< 1, Index >& dimensions )
{
   tnlAssert( dimensions[ 0 ] > 0,
              cerr << " dimensions[ 0 ] = " << dimensions[ 0 ] );
   this -> dimensions = dimensions;
   return tnlVector< Element, Device, Index > :: setSize( this -> dimensions[ 0 ] );
}

template< typename Element, typename Device, typename Index >
   template< typename MultiVector >
bool tnlMultiVector< 1, Element, Device, Index > :: setLike( const tnlMultiVector& multiVector )
{
   return setDimensions( multiVector. getDimensions() );
}

template< typename Element, typename Device, typename Index >
void tnlMultiVector< 1, Element, Device, Index > :: getDimensions( Index& xSize ) const
{
   xSize = this -> dimensions[ 0 ];
}

template< typename Element, typename Device, typename Index >
const tnlTuple< 1, Index >& tnlMultiVector< 1, Element, Device, Index > :: getDimensions() const
{
   return this -> dimensions;
}

template< typename Element, typename Device, typename Index >
Index tnlMultiVector< 1, Element, Device, Index > :: getElementIndex( const Index i ) const
{
   tnlAssert( i >= 0 && i < this -> dimensions[ 0 ],
              cerr << "i = " << i
                   << "this -> dimensions[ 0 ] " << this -> dimensions[ 0 ] );
   return i;
}

template< typename Element, typename Device, typename Index >
Element tnlMultiVector< 1, Element, Device, Index > :: getElement( const Index i ) const
{
   return tnlVector< Element, Device, Index > :: getElement( getElementIndex( i ) );
}

template< typename Element, typename Device, typename Index >
void tnlMultiVector< 1, Element, Device, Index > :: setElement( const Index i, Element value )
{
   tnlVector< Element, Device, Index > :: setElement( getElementIndex( i ), value );
}


template< typename Element, typename Device, typename Index >
Element& tnlMultiVector< 1, Element, Device, Index > :: operator()( const Index element )
{
   return tnlVector< Element, Device, Index > :: operator[]( getElementIndex( element ) );
}

template< typename Element, typename Device, typename Index >
const Element& tnlMultiVector< 1, Element, Device, Index > :: operator()( const Index element ) const
{
   return tnlVector< Element, Device, Index > :: operator[]( getElementIndex( element ) );
}

template< typename Element, typename Device, typename Index >
   template< typename MultiVector >
bool tnlMultiVector< 1, Element, Device, Index > :: operator == ( const MultiVector& Vector ) const
{
   // TODO: Static assert on dimensions
   tnlAssert( this -> getDimensions() == Vector. getDimensions(),
              cerr << "You are attempting to compare two Vectors with different dimensions." << endl
                   << "First Vector name is " << this -> getName()
                   << " dimensions are ( " << this -> getDimensions() << " )" << endl
                   << "Second Vector is " << Vector. getName()
                   << " dimensions are ( " << Vector. getDimensions() << " )" << endl; );
   return tnlVector< Element, Device, Index > :: operator == ( Vector );
}

template< typename Element, typename Device, typename Index >
   template< typename MultiVector >
bool tnlMultiVector< 1, Element, Device, Index > :: operator != ( const MultiVector& Vector ) const
{
   return ! ( (* this ) == Vector );
}

template< typename Element, typename Device, typename Index >
tnlMultiVector< 1, Element, Device, Index >&
   tnlMultiVector< 1, Element, Device, Index > :: operator = ( const tnlMultiVector< 1, Element, Device, Index >& Vector )
{
   // TODO: Static assert on dimensions
   tnlAssert( this -> getDimensions() == Vector. getDimensions(),
              cerr << "You are attempting to assign two Vectors with different dimensions." << endl
                   << "First Vector name is " << this -> getName()
                   << " dimensions are ( " << this -> getDimensions() << " )" << endl
                   << "Second Vector is " << Vector. getName()
                   << " dimensions are ( " << Vector. getDimensions() << " )" << endl; );
   tnlVector< Element, Device, Index > :: operator = ( Vector );
   return ( *this );
}

template< typename Element, typename Device, typename Index >
   template< typename MultiVector >
tnlMultiVector< 1, Element, Device, Index >&
   tnlMultiVector< 1, Element, Device, Index > :: operator = ( const MultiVector& Vector )
{
   // TODO: Static assert on dimensions
   tnlAssert( this -> getDimensions() == Vector. getDimensions(),
              cerr << "You are attempting to assign two Vectors with different dimensions." << endl
                   << "First Vector name is " << this -> getName()
                   << " dimensions are ( " << this -> getDimensions() << " )" << endl
                   << "Second Vector is " << Vector. getName()
                   << " dimensions are ( " << Vector. getDimensions() << " )" << endl; );
   tnlVector< Element, Device, Index > :: operator = ( Vector );
   return ( *this );
}

template< typename Element, typename Device, typename Index >
bool tnlMultiVector< 1, Element, Device, Index > :: save( tnlFile& file ) const
{
   if( ! tnlVector< Element, Device, Index > :: save( file ) )
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

template< typename Element, typename Device, typename Index >
bool tnlMultiVector< 1, Element, Device, Index > :: load( tnlFile& file )
{
   if( ! tnlVector< Element, Device, Index > :: load( file ) )
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

template< typename Element, typename Device, typename Index >
ostream& operator << ( ostream& str, const tnlMultiVector< 1, Element, Device, Index >& Vector )
{
   for( Index i = 0; i < Vector. getDimensions()[ 0 ]; i ++ )
   {
      str << Vector. getElement( i ) << " ";
   }
   return str;
}

template< typename Element, typename Device, typename Index >
bool tnlMultiVector< 1, Element, Device, Index > :: save( const tnlString& fileName ) const
{
   return tnlObject :: save( fileName );
}

template< typename Element, typename Device, typename Index >
bool tnlMultiVector< 1, Element, Device, Index > :: load( const tnlString& fileName )
{
   return tnlObject :: load( fileName );
}

#endif /* TNLMULTIVECTOR1D_IMPL_H_ */
