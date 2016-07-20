/***************************************************************************
                          tnlMultiVector1D_impl.h  -  description
                             -------------------
    begin                : Nov 13, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Vectors {   

template< typename Real, typename Device, typename Index >
tnlMultiVector< 1, Real, Device, Index > :: tnlMultiVector()
{
}

template< typename Real, typename Device, typename Index >
String tnlMultiVector< 1, Real, Device, Index > :: getType()
{
   return String( "tnlMultiVector< ") +
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
String tnlMultiVector< 1, Real, Device, Index > :: getTypeVirtual() const
{
   return this->getType();
};

template< typename Real,
          typename Device,
          typename Index >
String tnlMultiVector< 1, Real, Device, Index > :: getSerializationType()
{
   return HostType::getType();
};

template< typename Real,
          typename Device,
          typename Index >
String tnlMultiVector< 1, Real, Device, Index > :: getSerializationTypeVirtual() const
{
   return this->getSerializationType();
};

template< typename Real, typename Device, typename Index >
bool tnlMultiVector< 1, Real, Device, Index > :: setDimensions( const Index iSize )
{
   Assert( iSize > 0,
              std::cerr << "iSize = " << iSize );
   dimensions[ 0 ] = iSize;
   return tnlVector< Real, Device, Index > :: setSize( iSize );
}

template< typename Real, typename Device, typename Index >
bool tnlMultiVector< 1, Real, Device, Index > :: setDimensions( const tnlStaticVector< Dimensions, Index >& dimensions )
{
   Assert( dimensions[ 0 ] > 0,
              std::cerr << " dimensions[ 0 ] = " << dimensions[ 0 ] );
   this->dimensions = dimensions;
   return tnlVector< Real, Device, Index > :: setSize( this->dimensions[ 0 ] );
}

template< typename Real, typename Device, typename Index >
   template< typename MultiVector >
bool tnlMultiVector< 1, Real, Device, Index > :: setLike( const tnlMultiVector& multiVector )
{
   return setDimensions( multiVector. getDimensions() );
}

template< typename Real, typename Device, typename Index >
void tnlMultiVector< 1, Real, Device, Index > :: getDimensions( Index& xSize ) const
{
   xSize = this->dimensions[ 0 ];
}

template< typename Real, typename Device, typename Index >
const tnlStaticVector< 1, Index >& tnlMultiVector< 1, Real, Device, Index > :: getDimensions() const
{
   return this->dimensions;
}

template< typename Real, typename Device, typename Index >
Index tnlMultiVector< 1, Real, Device, Index > :: getElementIndex( const Index i ) const
{
   Assert( i >= 0 && i < this->dimensions[ 0 ],
              std::cerr << "i = " << i
                   << "this->dimensions[ 0 ] " << this->dimensions[ 0 ] );
   return i;
}

template< typename Real, typename Device, typename Index >
Real tnlMultiVector< 1, Real, Device, Index > :: getElement( const Index i ) const
{
   return tnlVector< Real, Device, Index > :: getElement( getElementIndex( i ) );
}

template< typename Real, typename Device, typename Index >
void tnlMultiVector< 1, Real, Device, Index > :: setElement( const Index i, Real value )
{
   tnlVector< Real, Device, Index > :: setElement( getElementIndex( i ), value );
}


template< typename Real, typename Device, typename Index >
Real& tnlMultiVector< 1, Real, Device, Index > :: operator()( const Index element )
{
   return tnlVector< Real, Device, Index > :: operator[]( getElementIndex( element ) );
}

template< typename Real, typename Device, typename Index >
const Real& tnlMultiVector< 1, Real, Device, Index > :: operator()( const Index element ) const
{
   return tnlVector< Real, Device, Index > :: operator[]( getElementIndex( element ) );
}

template< typename Real, typename Device, typename Index >
   template< typename MultiVector >
bool tnlMultiVector< 1, Real, Device, Index > :: operator == ( const MultiVector& Vector ) const
{
   // TODO: Static assert on dimensions
   Assert( this->getDimensions() == Vector. getDimensions(),
              std::cerr << "You are attempting to compare two Vectors with different dimensions." << std::endl
                   << "First Vector name dimensions are ( " << this->getDimensions() << " )" << std::endl
                   << "Second Vector dimensions are ( " << Vector. getDimensions() << " )" << std::endl; );
   return tnlVector< Real, Device, Index > :: operator == ( Vector );
}

template< typename Real, typename Device, typename Index >
   template< typename MultiVector >
bool tnlMultiVector< 1, Real, Device, Index > :: operator != ( const MultiVector& Vector ) const
{
   return ! ( (* this ) == Vector );
}

template< typename Real, typename Device, typename Index >
tnlMultiVector< 1, Real, Device, Index >&
   tnlMultiVector< 1, Real, Device, Index > :: operator = ( const tnlMultiVector< 1, Real, Device, Index >& Vector )
{
   // TODO: Static assert on dimensions
   Assert( this->getDimensions() == Vector. getDimensions(),
              std::cerr << "You are attempting to assign two Vectors with different dimensions." << std::endl
                   << "First vector dimensions are ( " << this->getDimensions() << " )" << std::endl
                   << "Second vector dimensions are ( " << Vector. getDimensions() << " )" << std::endl; );
   tnlVector< Real, Device, Index > :: operator = ( Vector );
   return ( *this );
}

template< typename Real, typename Device, typename Index >
   template< typename MultiVector >
tnlMultiVector< 1, Real, Device, Index >&
   tnlMultiVector< 1, Real, Device, Index > :: operator = ( const MultiVector& Vector )
{
   // TODO: Static assert on dimensions
   Assert( this->getDimensions() == Vector. getDimensions(),
              std::cerr << "You are attempting to assign two Vectors with different dimensions." << std::endl
                   << "First vector dimensions are ( " << this->getDimensions() << " )" << std::endl
                   << "Second vector dimensions are ( " << Vector. getDimensions() << " )" << std::endl; );
   tnlVector< Real, Device, Index > :: operator = ( Vector );
   return ( *this );
}

template< typename Real, typename Device, typename Index >
bool tnlMultiVector< 1, Real, Device, Index > :: save( File& file ) const
{
   if( ! tnlVector< Real, Device, Index > :: save( file ) )
   {
      std::cerr << "I was not able to write the tnlVector of tnlMultiVector." << std::endl;
      return false;
   }
   if( ! dimensions. save( file ) )
   {
      std::cerr << "I was not able to write the dimensions of tnlMultiVector." << std::endl;
      return false;
   }
   return true;
}

template< typename Real, typename Device, typename Index >
bool tnlMultiVector< 1, Real, Device, Index > :: load( File& file )
{
   if( ! tnlVector< Real, Device, Index > :: load( file ) )
   {
      std::cerr << "I was not able to read the tnlVector of tnlMultiVector." << std::endl;
      return false;
   }
   if( ! dimensions. load( file ) )
   {
      std::cerr << "I was not able to read the dimensions of tnlMultiVector." << std::endl;
      return false;
   }
   return true;
}

template< typename Real, typename Device, typename Index >
std::ostream& operator << ( std::ostream& str, const tnlMultiVector< 1, Real, Device, Index >& Vector )
{
   for( Index i = 0; i < Vector. getDimensions()[ 0 ]; i ++ )
   {
      str << Vector. getElement( i ) << " ";
   }
   return str;
}

template< typename Real, typename Device, typename Index >
bool tnlMultiVector< 1, Real, Device, Index > :: save( const String& fileName ) const
{
   return Object :: save( fileName );
}

template< typename Real, typename Device, typename Index >
bool tnlMultiVector< 1, Real, Device, Index > :: load( const String& fileName )
{
   return Object :: load( fileName );
}

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

#ifdef INSTANTIATE_FLOAT
extern template class tnlMultiVector< 1, float,  tnlHost, int >;
#endif
extern template class tnlMultiVector< 1, double, tnlHost, int >;
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
extern template class tnlMultiVector< 1, float,  tnlHost, long int >;
#endif
extern template class tnlMultiVector< 1, double, tnlHost, long int >;
#endif

#ifdef HAVE_CUDA
/*#ifdef INSTANTIATE_FLOAT
extern template class tnlMultiVector< 1, float,  tnlCuda, int >;
#endif
extern template class tnlMultiVector< 1, double, tnlCuda, int >;
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
extern template class tnlMultiVector< 1, float,  tnlCuda, long int >;
#endif
extern template class tnlMultiVector< 1, double, tnlCuda, long int >;
#endif*/
#endif

#endif

} // namespace Vectors
} // namespace TNL
