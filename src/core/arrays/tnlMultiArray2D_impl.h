/***************************************************************************
                          tnlMultiArray2D_impl.h  -  description
                             -------------------
    begin                : Nov 13, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {

template< typename Element, typename Device, typename Index >
tnlMultiArray< 2, Element, Device, Index > :: tnlMultiArray()
{
}

template< typename Element, typename Device, typename Index >
tnlString tnlMultiArray< 2, Element, Device, Index > :: getType()
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
tnlString tnlMultiArray< 2, Element, Device, Index > :: getTypeVirtual() const
{
   return this->getType();
};

template< typename Element,
          typename Device,
          typename Index >
tnlString tnlMultiArray< 2, Element, Device, Index > :: getSerializationType()
{
   return HostType::getType();
};

template< typename Element,
          typename Device,
          typename Index >
tnlString tnlMultiArray< 2, Element, Device, Index > :: getSerializationTypeVirtual() const
{
   return this->getSerializationType();
};

template< typename Element, typename Device, typename Index >
bool tnlMultiArray< 2, Element, Device, Index > :: setDimensions( const Index jSize,
                                                                  const Index iSize )
{
   tnlAssert( iSize > 0 && jSize > 0,
              cerr << "iSize = " << iSize
                   << "jSize = " << jSize );

   dimensions[ 0 ] = iSize;
   dimensions[ 1 ] = jSize;
   return tnlArray< Element, Device, Index > :: setSize( iSize * jSize );
}

template< typename Element, typename Device, typename Index >
bool tnlMultiArray< 2, Element, Device, Index > :: setDimensions( const tnlStaticVector< 2, Index >& dimensions )
{
   tnlAssert( dimensions[ 0 ] > 0 && dimensions[ 1 ] > 0,
              cerr << "dimensions = " << dimensions );
   /****
    * Swap the dimensions in the tuple to be compatible with the previous method.
    */
   this->dimensions. x() = dimensions. y();
   this->dimensions. y() = dimensions. x();
   return tnlArray< Element, Device, Index > :: setSize( this->dimensions[ 1 ] * this->dimensions[ 0 ] );
}

template< typename Element, typename Device, typename Index >
   template< typename MultiArray >
bool tnlMultiArray< 2, Element, Device, Index > :: setLike( const MultiArray& multiArray )
{
   return setDimensions( multiArray. getDimensions() );
}

template< typename Element, typename Device, typename Index >
void tnlMultiArray< 2, Element, Device, Index >::reset()
{
   this->dimensions = tnlStaticVector< 2, Index >( ( Index ) 0 );
   tnlArray< Element, Device, Index >::reset();
}

template< typename Element, typename Device, typename Index >
__cuda_callable__
void tnlMultiArray< 2, Element, Device, Index > :: getDimensions( Index& jSize, Index& iSize ) const
{
   iSize = this->dimensions[ 0 ];
   jSize = this->dimensions[ 1 ];
}

template< typename Element, typename Device, typename Index >
__cuda_callable__
const tnlStaticVector< 2, Index >& tnlMultiArray< 2, Element, Device, Index > :: getDimensions() const
{
   return this->dimensions;
}

template< typename Element, typename Device, typename Index >
__cuda_callable__
Index tnlMultiArray< 2, Element, Device, Index > :: getElementIndex( const Index j, const Index i ) const
{
   tnlAssert( i >= 0 && i < this->dimensions[ 0 ] && j >= 0 && j < this->dimensions[ 1 ],
              cerr << "i = " << i << " j = " << j << " this->dimensions[ 0 ] = " <<  this->dimensions[ 0 ]
                   << " this->dimensions[ 1 ] = " << this->dimensions[ 1 ] );
   return j * this->dimensions[ 0 ] + i;
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
__cuda_callable__
Element& tnlMultiArray< 2, Element, Device, Index > :: operator()( const Index j, const Index i )
{
   return tnlArray< Element, Device, Index > :: operator[]( getElementIndex( j, i ) );
}

template< typename Element, typename Device, typename Index >
__cuda_callable__
const Element& tnlMultiArray< 2, Element, Device, Index > :: operator()( const Index j, const Index i ) const
{
   return tnlArray< Element, Device, Index > :: operator[]( getElementIndex( j, i ) );
}

template< typename Element, typename Device, typename Index >
   template< typename MultiArray >
bool tnlMultiArray< 2, Element, Device, Index > :: operator == ( const MultiArray& array ) const
{
   // TODO: Static assert on dimensions
   tnlAssert( this->getDimensions() == array. getDimensions(),
              cerr << "You are attempting to compare two arrays with different dimensions." << endl
                   << "First array dimensions are ( " << this->getDimensions() << " )" << endl
                   << "Second array dimensions are ( " << array. getDimensions() << " )" << endl; );
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
   tnlAssert( this->getDimensions() == array. getDimensions(),
              cerr << "You are attempting to assign two arrays with different dimensions." << endl
                   << "First array dimensions are ( " << this->getDimensions() << " )" << endl
                   << "Second array dimensions are ( " << array. getDimensions() << " )" << endl; );
   tnlArray< Element, Device, Index > :: operator = ( array );
   return ( *this );
}

template< typename Element, typename Device, typename Index >
   template< typename MultiArray >
tnlMultiArray< 2, Element, Device, Index >&
   tnlMultiArray< 2, Element, Device, Index > :: operator = ( const MultiArray& array )
{
   // TODO: Static assert on dimensions
   tnlAssert( this->getDimensions() == array. getDimensions(),
              cerr << "You are attempting to assign two arrays with different dimensions." << endl
                   << "First array dimensions are ( " << this->getDimensions() << " )" << endl
                   << "Second array dimensions are ( " << array. getDimensions() << " )" << endl; );
   tnlArray< Element, Device, Index > :: operator = ( array );
   return ( *this );
}

template< typename Element, typename Device, typename Index >
bool tnlMultiArray< 2, Element, Device, Index > :: save( tnlFile& file ) const
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
bool tnlMultiArray< 2, Element, Device, Index > :: load( tnlFile& file )
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
bool tnlMultiArray< 2, Element, Device, Index > :: save( const tnlString& fileName ) const
{
   return tnlObject :: save( fileName );
}

template< typename Element, typename Device, typename Index >
bool tnlMultiArray< 2, Element, Device, Index > :: load( const tnlString& fileName )
{
   return tnlObject :: load( fileName );
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

} // namespace TNL
