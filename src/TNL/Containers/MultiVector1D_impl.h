/***************************************************************************
                          MultiVector1D_impl.h  -  description
                             -------------------
    begin                : Nov 13, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Containers {   

template< typename Real, typename Device, typename Index >
MultiVector< 1, Real, Device, Index > :: MultiVector()
{
}

template< typename Real, typename Device, typename Index >
String MultiVector< 1, Real, Device, Index > :: getType()
{
   return String( "Containers::MultiVector< ") +
          String( Dimension ) +
          String( ", " ) +
          String( TNL::getType< Real >() ) +
          String( ", " ) +
          String( Device :: getDeviceType() ) +
          String( ", " ) +
          String( TNL::getType< Index >() ) +
          String( " >" );
}

template< typename Real,
          typename Device,
          typename Index >
String MultiVector< 1, Real, Device, Index > :: getTypeVirtual() const
{
   return this->getType();
};

template< typename Real,
          typename Device,
          typename Index >
String MultiVector< 1, Real, Device, Index > :: getSerializationType()
{
   return HostType::getType();
};

template< typename Real,
          typename Device,
          typename Index >
String MultiVector< 1, Real, Device, Index > :: getSerializationTypeVirtual() const
{
   return this->getSerializationType();
};

template< typename Real, typename Device, typename Index >
void MultiVector< 1, Real, Device, Index > :: setDimensions( const Index iSize )
{
   TNL_ASSERT( iSize > 0,
              std::cerr << "iSize = " << iSize );
   dimensions[ 0 ] = iSize;
   Vector< Real, Device, Index > :: setSize( iSize );
}

template< typename Real, typename Device, typename Index >
void MultiVector< 1, Real, Device, Index > :: setDimensions( const StaticVector< Dimension, Index >& dimensions )
{
   TNL_ASSERT( dimensions[ 0 ] > 0,
              std::cerr << " dimensions[ 0 ] = " << dimensions[ 0 ] );
   this->dimensions = dimensions;
   Vector< Real, Device, Index > :: setSize( this->dimensions[ 0 ] );
}

template< typename Real, typename Device, typename Index >
   template< typename MultiVectorT >
void MultiVector< 1, Real, Device, Index > :: setLike( const MultiVectorT& multiVector )
{
   setDimensions( multiVector. getDimensions() );
}

template< typename Real, typename Device, typename Index >
void MultiVector< 1, Real, Device, Index > :: getDimensions( Index& xSize ) const
{
   xSize = this->dimensions[ 0 ];
}

template< typename Real, typename Device, typename Index >
const StaticVector< 1, Index >& MultiVector< 1, Real, Device, Index > :: getDimensions() const
{
   return this->dimensions;
}

template< typename Real, typename Device, typename Index >
Index MultiVector< 1, Real, Device, Index > :: getElementIndex( const Index i ) const
{
   TNL_ASSERT( i >= 0 && i < this->dimensions[ 0 ],
              std::cerr << "i = " << i
                   << "this->dimensions[ 0 ] " << this->dimensions[ 0 ] );
   return i;
}

template< typename Real, typename Device, typename Index >
Real MultiVector< 1, Real, Device, Index > :: getElement( const Index i ) const
{
   return Vector< Real, Device, Index > :: getElement( getElementIndex( i ) );
}

template< typename Real, typename Device, typename Index >
void MultiVector< 1, Real, Device, Index > :: setElement( const Index i, Real value )
{
   Vector< Real, Device, Index > :: setElement( getElementIndex( i ), value );
}


template< typename Real, typename Device, typename Index >
Real& MultiVector< 1, Real, Device, Index > :: operator()( const Index element )
{
   return Vector< Real, Device, Index > :: operator[]( getElementIndex( element ) );
}

template< typename Real, typename Device, typename Index >
const Real& MultiVector< 1, Real, Device, Index > :: operator()( const Index element ) const
{
   return Vector< Real, Device, Index > :: operator[]( getElementIndex( element ) );
}

template< typename Real, typename Device, typename Index >
   template< typename MultiVectorT >
bool MultiVector< 1, Real, Device, Index > :: operator == ( const MultiVectorT& vector ) const
{
   // TODO: Static assert on dimensions
   TNL_ASSERT( this->getDimensions() == vector. getDimensions(),
              std::cerr << "You are attempting to compare two Vectors with different dimensions." << std::endl
                   << "First Vector name dimensions are ( " << this->getDimensions() << " )" << std::endl
                   << "Second Vector dimensions are ( " << vector. getDimensions() << " )" << std::endl; );
   return Vector< Real, Device, Index > :: operator == ( vector );
}

template< typename Real, typename Device, typename Index >
   template< typename MultiVectorT >
bool MultiVector< 1, Real, Device, Index > :: operator != ( const MultiVectorT& vector ) const
{
   return ! ( (* this ) == vector );
}

template< typename Real, typename Device, typename Index >
MultiVector< 1, Real, Device, Index >&
   MultiVector< 1, Real, Device, Index > :: operator = ( const MultiVector< 1, Real, Device, Index >& vector )
{
   // TODO: Static assert on dimensions
   TNL_ASSERT( this->getDimensions() == vector. getDimensions(),
              std::cerr << "You are attempting to assign two Vectors with different dimensions." << std::endl
                   << "First vector dimensions are ( " << this->getDimensions() << " )" << std::endl
                   << "Second vector dimensions are ( " << vector. getDimensions() << " )" << std::endl; );
   Vector< Real, Device, Index > :: operator = ( vector );
   return ( *this );
}

template< typename Real, typename Device, typename Index >
   template< typename MultiVectorT >
MultiVector< 1, Real, Device, Index >&
   MultiVector< 1, Real, Device, Index > :: operator = ( const MultiVectorT& vector )
{
   // TODO: Static assert on dimensions
   TNL_ASSERT( this->getDimensions() == vector. getDimensions(),
              std::cerr << "You are attempting to assign two Vectors with different dimensions." << std::endl
                   << "First vector dimensions are ( " << this->getDimensions() << " )" << std::endl
                   << "Second vector dimensions are ( " << vector. getDimensions() << " )" << std::endl; );
   Vector< Real, Device, Index > :: operator = ( vector );
   return ( *this );
}

template< typename Real, typename Device, typename Index >
bool MultiVector< 1, Real, Device, Index > :: save( File& file ) const
{
   if( ! Vector< Real, Device, Index > :: save( file ) )
   {
      std::cerr << "I was not able to write the Vector of MultiVector." << std::endl;
      return false;
   }
   if( ! dimensions. save( file ) )
   {
      std::cerr << "I was not able to write the dimensions of MultiVector." << std::endl;
      return false;
   }
   return true;
}

template< typename Real, typename Device, typename Index >
bool MultiVector< 1, Real, Device, Index > :: load( File& file )
{
   if( ! Vector< Real, Device, Index > :: load( file ) )
   {
      std::cerr << "I was not able to read the Vector of MultiVector." << std::endl;
      return false;
   }
   if( ! dimensions. load( file ) )
   {
      std::cerr << "I was not able to read the dimensions of MultiVector." << std::endl;
      return false;
   }
   return true;
}

template< typename Real, typename Device, typename Index >
std::ostream& operator << ( std::ostream& str, const MultiVector< 1, Real, Device, Index >& Vector )
{
   for( Index i = 0; i < Vector. getDimensions()[ 0 ]; i ++ )
   {
      str << Vector. getElement( i ) << " ";
   }
   return str;
}

template< typename Real, typename Device, typename Index >
bool MultiVector< 1, Real, Device, Index > :: save( const String& fileName ) const
{
   return Object :: save( fileName );
}

template< typename Real, typename Device, typename Index >
bool MultiVector< 1, Real, Device, Index > :: load( const String& fileName )
{
   return Object :: load( fileName );
}

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

#ifdef INSTANTIATE_FLOAT
extern template class MultiVector< 1, float,  Devices::Host, int >;
#endif
extern template class MultiVector< 1, double, Devices::Host, int >;
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
extern template class MultiVector< 1, float,  Devices::Host, long int >;
#endif
extern template class MultiVector< 1, double, Devices::Host, long int >;
#endif

#ifdef HAVE_CUDA
/*#ifdef INSTANTIATE_FLOAT
extern template class MultiVector< 1, float,  Devices::Cuda, int >;
#endif
extern template class MultiVector< 1, double, Devices::Cuda, int >;
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
extern template class MultiVector< 1, float,  Devices::Cuda, long int >;
#endif
extern template class MultiVector< 1, double, Devices::Cuda, long int >;
#endif*/
#endif

#endif

} // namespace Containers
} // namespace TNL
