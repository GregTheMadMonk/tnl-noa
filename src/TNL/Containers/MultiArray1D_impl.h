/***************************************************************************
                          MultiArray1D_impl.h  -  description
                             -------------------
    begin                : Nov 13, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

namespace TNL {
namespace Containers {   

template< typename Element, typename Device, typename Index >
MultiArray< 1, Element, Device, Index > :: MultiArray()
{
}

template< typename Element, typename Device, typename Index >
String MultiArray< 1, Element, Device, Index > :: getType()
{
   return String( "Containers::MultiArray< ") +
          String( Dimension ) +
          String( ", " ) +
          String( TNL::getType< Element >() ) +
          String( ", " ) +
          String( Device :: getDeviceType() ) +
          String( ", " ) +
          String( TNL::getType< Index >() ) +
          String( " >" );
}

template< typename Element,
          typename Device,
          typename Index >
String MultiArray< 1, Element, Device, Index > :: getTypeVirtual() const
{
   return this->getType();
};

template< typename Element,
          typename Device,
          typename Index >
String MultiArray< 1, Element, Device, Index > :: getSerializationType()
{
   return HostType::getType();
};

template< typename Element,
          typename Device,
          typename Index >
String MultiArray< 1, Element, Device, Index > :: getSerializationTypeVirtual() const
{
   return this->getSerializationType();
};

template< typename Element, typename Device, typename Index >
bool MultiArray< 1, Element, Device, Index > :: setDimensions( const Index iSize )
{
   TNL_ASSERT( iSize > 0,
              std::cerr << "iSize = " << iSize );
   dimensions[ 0 ] = iSize;
   return Array< Element, Device, Index >::setSize( iSize );
}

template< typename Element, typename Device, typename Index >
bool MultiArray< 1, Element, Device, Index > :: setDimensions( const Containers::StaticVector< 1, Index >& dimensions )
{
   TNL_ASSERT( dimensions[ 0 ] > 0,
              std::cerr << " dimensions[ 0 ] = " << dimensions[ 0 ] );
   this->dimensions = dimensions;
   return Array< Element, Device, Index >::setSize( this->dimensions[ 0 ] );
}

template< typename Element, typename Device, typename Index >
   template< typename MultiArrayT >
bool MultiArray< 1, Element, Device, Index > :: setLike( const MultiArrayT& multiArray )
{
   return setDimensions( multiArray. getDimensions() );
}

template< typename Element, typename Device, typename Index >
void MultiArray< 1, Element, Device, Index >::reset()
{
   this->dimensions = Containers::StaticVector< 1, Index >( ( Index ) 0 );
   Array< Element, Device, Index >::reset();
}

template< typename Element, typename Device, typename Index >
__cuda_callable__
void MultiArray< 1, Element, Device, Index > :: getDimensions( Index& xSize ) const
{
   xSize = this->dimensions[ 0 ];
}

template< typename Element, typename Device, typename Index >
__cuda_callable__
const Containers::StaticVector< 1, Index >& MultiArray< 1, Element, Device, Index > :: getDimensions() const
{
   return this->dimensions;
}

template< typename Element, typename Device, typename Index >
__cuda_callable__
Index MultiArray< 1, Element, Device, Index > :: getElementIndex( const Index i ) const
{
   TNL_ASSERT( i >= 0 && i < this->dimensions[ 0 ],
              std::cerr << "i = " << i << " this->dimensions[ 0 ] = " <<  this->dimensions[ 0 ] );
   return i;
}

template< typename Element, typename Device, typename Index >
Element MultiArray< 1, Element, Device, Index > :: getElement( const Index i ) const
{
   return Array< Element, Device, Index > :: getElement( getElementIndex( i ) );
}

template< typename Element, typename Device, typename Index >
void MultiArray< 1, Element, Device, Index > :: setElement( const Index i, Element value )
{
   Array< Element, Device, Index > :: setElement( getElementIndex( i ), value );
}

template< typename Element, typename Device, typename Index >
__cuda_callable__
Element& MultiArray< 1, Element, Device, Index > :: operator()( const Index element )
{
   return Array< Element, Device, Index > :: operator[]( getElementIndex( element ) );
}

template< typename Element, typename Device, typename Index >
__cuda_callable__
const Element& MultiArray< 1, Element, Device, Index > :: operator()( const Index element ) const
{
   return Array< Element, Device, Index > :: operator[]( getElementIndex( element ) );
}

template< typename Element, typename Device, typename Index >
   template< typename MultiArrayT >
bool MultiArray< 1, Element, Device, Index > :: operator == ( const MultiArrayT& array ) const
{
   // TODO: Static assert on dimensions
   TNL_ASSERT( this->getDimensions() == array. getDimensions(),
              std::cerr << "You are attempting to compare two arrays with different dimensions." << std::endl
                   << "First array dimensions are ( " << this->getDimensions() << " )" << std::endl
                   << "Second array dimensions are ( " << array. getDimensions() << " )" << std::endl; );
   return Array< Element, Device, Index > :: operator == ( array );
}

template< typename Element, typename Device, typename Index >
   template< typename MultiArrayT >
bool MultiArray< 1, Element, Device, Index > :: operator != ( const MultiArrayT& array ) const
{
   return ! ( (* this ) == array );
}

template< typename Element, typename Device, typename Index >
MultiArray< 1, Element, Device, Index >&
   MultiArray< 1, Element, Device, Index > :: operator = ( const MultiArray< 1, Element, Device, Index >& array )
{
   // TODO: Static assert on dimensions
   TNL_ASSERT( this->getDimensions() == array. getDimensions(),
              std::cerr << "You are attempting to assign two arrays with different dimensions." << std::endl
                   << "First array dimensions are ( " << this->getDimensions() << " )" << std::endl
                   << "Second array dimensions are ( " << array. getDimensions() << " )" << std::endl; );
   Array< Element, Device, Index > :: operator = ( array );
   return ( *this );
}

template< typename Element, typename Device, typename Index >
   template< typename MultiArrayT >
MultiArray< 1, Element, Device, Index >&
   MultiArray< 1, Element, Device, Index > :: operator = ( const MultiArrayT& array )
{
   // TODO: Static assert on dimensions
   TNL_ASSERT( this->getDimensions() == array. getDimensions(),
              std::cerr << "You are attempting to assign two arrays with different dimensions." << std::endl
                   << "First array dimensions are ( " << this->getDimensions() << " )" << std::endl
                   << "Second array dimensions are ( " << array. getDimensions() << " )" << std::endl; );
   Array< Element, Device, Index > :: operator = ( array );
   return ( *this );
}

template< typename Element, typename Device, typename Index >
bool MultiArray< 1, Element, Device, Index > :: save( File& file ) const
{
   if( ! Array< Element, Device, Index > :: save( file ) )
   {
      std::cerr << "I was not able to write the Array of MultiArray." << std::endl;
      return false;
   }
   if( ! dimensions. save( file ) )
   {
      std::cerr << "I was not able to write the dimensions of MultiArray." << std::endl;
      return false;
   }
   return true;
}

template< typename Element, typename Device, typename Index >
bool MultiArray< 1, Element, Device, Index > :: load( File& file )
{
   if( ! Array< Element, Device, Index > :: load( file ) )
   {
      std::cerr << "I was not able to read the Array of MultiArray." << std::endl;
      return false;
   }
   if( ! dimensions. load( file ) )
   {
      std::cerr << "I was not able to read the dimensions of MultiArray." << std::endl;
      return false;
   }
   return true;
}

template< typename Element, typename Device, typename Index >
bool MultiArray< 1, Element, Device, Index > :: save( const String& fileName ) const
{
   return Object :: save( fileName );
}

template< typename Element, typename Device, typename Index >
bool MultiArray< 1, Element, Device, Index > :: load( const String& fileName )
{
   return Object :: load( fileName );
}

template< typename Element, typename Device, typename Index >
std::ostream& operator << ( std::ostream& str, const MultiArray< 1, Element, Device, Index >& array )
{
   for( Index i = 0; i < array. getDimensions()[ 0 ]; i ++ )
   {
      str << array. getElement( i ) << " ";
   }
   return str;
}

} // namespace Containers
} // namespace TNL
