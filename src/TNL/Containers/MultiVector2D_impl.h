/***************************************************************************
                          MultiVector2D_impl.h  -  description
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
MultiVector< 2, Real, Device, Index > :: MultiVector()
{
}

template< typename Real, typename Device, typename Index >
String MultiVector< 2, Real, Device, Index > :: getType()
{
   return String( "Containers::MultiVector< ") +
          String( Dimensions ) +
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
String MultiVector< 2, Real, Device, Index > :: getTypeVirtual() const
{
   return this->getType();
};

template< typename Real,
          typename Device,
          typename Index >
String MultiVector< 2, Real, Device, Index > :: getSerializationType()
{
   return HostType::getType();
};

template< typename Real,
          typename Device,
          typename Index >
String MultiVector< 2, Real, Device, Index > :: getSerializationTypeVirtual() const
{
   return this->getSerializationType();
};

template< typename Real, typename Device, typename Index >
bool MultiVector< 2, Real, Device, Index > :: setDimensions( const Index jSize,
                                                                       const Index iSize )
{
   Assert( iSize > 0 && jSize > 0,
              std::cerr << "iSize = " << iSize
                   << "jSize = " << jSize );

   dimensions[ 0 ] = iSize;
   dimensions[ 1 ] = jSize;
   return Vector< Real, Device, Index > :: setSize( iSize * jSize );
}

template< typename Real, typename Device, typename Index >
bool MultiVector< 2, Real, Device, Index > :: setDimensions( const StaticVector< 2, Index >& dimensions )
{
   Assert( dimensions[ 0 ] > 0 && dimensions[ 1 ] > 0,
              std::cerr << "dimensions = " << dimensions );
   this->dimensions = dimensions;
   return Vector< Real, Device, Index > :: setSize( this->dimensions[ 1 ] * this->dimensions[ 0 ] );
}

template< typename Real, typename Device, typename Index >
   template< typename MultiVectorT >
bool MultiVector< 2, Real, Device, Index > :: setLike( const MultiVectorT& multiVector )
{
   return setDimensions( multiVector. getDimensions() );
}

template< typename Real, typename Device, typename Index >
void MultiVector< 2, Real, Device, Index > :: getDimensions( Index& jSize, Index& iSize ) const
{
   iSize = this->dimensions[ 0 ];
   jSize = this->dimensions[ 1 ];
}

template< typename Real, typename Device, typename Index >
const StaticVector< 2, Index >& MultiVector< 2, Real, Device, Index > :: getDimensions() const
{
   return this->dimensions;
}

template< typename Real, typename Device, typename Index >
Index MultiVector< 2, Real, Device, Index > :: getElementIndex( const Index j, const Index i ) const
{
   Assert( i >= 0 && i < this->dimensions[ 0 ] && j >= 0 && j < this->dimensions[ 1 ],
              std::cerr << "i = " << i
                   << "j = " << j
                   << "this->dimensions[ 0 ] = " << this->dimensions[ 0 ]
                   << "this->dimensions[ 1 ] = " << this->dimensions[ 1 ] );
   return j * this->dimensions[ 0 ] + i;
}

template< typename Real, typename Device, typename Index >
Real MultiVector< 2, Real, Device, Index > :: getElement( const Index j, const Index i ) const
{
   return Vector< Real, Device, Index > :: getElement( getElementIndex( j, i ) );
}

template< typename Real, typename Device, typename Index >
void MultiVector< 2, Real, Device, Index > :: setElement( const Index j, const Index i, Real value )
{
   Vector< Real, Device, Index > :: setElement( getElementIndex( j, i ), value );
}


template< typename Real, typename Device, typename Index >
Real& MultiVector< 2, Real, Device, Index > :: operator()( const Index j, const Index i )
{
   return Vector< Real, Device, Index > :: operator[]( getElementIndex( j, i ) );
}

template< typename Real, typename Device, typename Index >
const Real& MultiVector< 2, Real, Device, Index > :: operator()( const Index j, const Index i ) const
{
   return Vector< Real, Device, Index > :: operator[]( getElementIndex( j, i ) );
}

template< typename Real, typename Device, typename Index >
   template< typename MultiVectorT >
bool MultiVector< 2, Real, Device, Index > :: operator == ( const MultiVectorT& vector ) const
{
   // TODO: Static assert on dimensions
   Assert( this->getDimensions() == vector. getDimensions(),
              std::cerr << "You are attempting to compare two Vectors with different dimensions." << std::endl
                   << "First vector dimensions are ( " << this->getDimensions() << " )" << std::endl
                   << "Second vector dimensions are ( " << vector. getDimensions() << " )" << std::endl; );
   return Vector< Real, Device, Index > :: operator == ( vector );
}

template< typename Real, typename Device, typename Index >
   template< typename MultiVectorT >
bool MultiVector< 2, Real, Device, Index > :: operator != ( const MultiVectorT& vector ) const
{
   return ! ( (* this ) == vector );
}

template< typename Real, typename Device, typename Index >
MultiVector< 2, Real, Device, Index >&
   MultiVector< 2, Real, Device, Index > :: operator = ( const MultiVector< 2, Real, Device, Index >& vector )
{
   // TODO: Static assert on dimensions
   Assert( this->getDimensions() == vector. getDimensions(),
              std::cerr << "You are attempting to assign two Vectors with different dimensions." << std::endl
                   << "First vector dimensions are ( " << this->getDimensions() << " )" << std::endl
                   << "Second vector dimensions are ( " << vector. getDimensions() << " )" << std::endl; );
   Vector< Real, Device, Index > :: operator = ( vector );
   return ( *this );
}

template< typename Real, typename Device, typename Index >
   template< typename MultiVectorT >
MultiVector< 2, Real, Device, Index >&
   MultiVector< 2, Real, Device, Index > :: operator = ( const MultiVectorT& vector )
{
   // TODO: Static assert on dimensions
   Assert( this->getDimensions() == vector. getDimensions(),
              std::cerr << "You are attempting to assign two Vectors with different dimensions." << std::endl
                   << "First vector dimensions are ( " << this->getDimensions() << " )" << std::endl
                   << "Second vector dimensions are ( " << vector. getDimensions() << " )" << std::endl; );
   Vector< Real, Device, Index > :: operator = ( vector );
   return ( *this );
}

template< typename Real, typename Device, typename Index >
bool MultiVector< 2, Real, Device, Index > :: save( File& file ) const
{
   if( ! Vector< Real, Device, Index > :: save( file ) )
   {
      std::cerr << "I was not able to write the Vector of MultiVector."  << std::endl;
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
bool MultiVector< 2, Real, Device, Index > :: load( File& file )
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
bool MultiVector< 2, Real, Device, Index > :: save( const String& fileName ) const
{
   return Object :: save( fileName );
}

template< typename Real, typename Device, typename Index >
bool MultiVector< 2, Real, Device, Index > :: load( const String& fileName )
{
   return Object :: load( fileName );
}

template< typename Real, typename Device, typename Index >
std::ostream& operator << ( std::ostream& str, const MultiVector< 2, Real, Device, Index >& Vector )
{
   for( Index j = 0; j < Vector. getDimensions()[ 1 ]; j ++ )
   {
      for( Index i = 0; i < Vector. getDimensions()[ 0 ]; i ++ )
      {
         str << Vector. getElement( j, i ) << " ";
      }
      str << std::endl;
   }
   return str;
}

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

#ifdef INSTANTIATE_FLOAT
extern template class MultiVector< 2, float,  Devices::Host, int >;
#endif
extern template class MultiVector< 2, double, Devices::Host, int >;
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
extern template class MultiVector< 2, float,  Devices::Host, long int >;
#endif
extern template class MultiVector< 2, double, Devices::Host, long int >;
#endif

#ifdef HAVE_CUDA
/*#ifdef INSTANTIATE_FLOAT
extern template class MultiVector< 2, float,  Devices::Cuda, int >;
#endif
extern template class MultiVector< 2, double, Devices::Cuda, int >;
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
extern template class MultiVector< 2, float,  Devices::Cuda, long int >;
#endif
extern template class MultiVector< 2, double, Devices::Cuda, long int >;
#endif*/
#endif

#endif

} // namespace Containers
} // namespace TNL
