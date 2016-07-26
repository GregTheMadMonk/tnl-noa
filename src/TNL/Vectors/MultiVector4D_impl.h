/***************************************************************************
                          tnlMultiVector4D_impl.h  -  description
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
tnlMultiVector< 4, Real, Device, Index > :: tnlMultiVector()
{
}

template< typename Real, typename Device, typename Index >
String tnlMultiVector< 4, Real, Device, Index > :: getType()
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
String tnlMultiVector< 4, Real, Device, Index > :: getTypeVirtual() const
{
   return this->getType();
};

template< typename Real,
          typename Device,
          typename Index >
String tnlMultiVector< 4, Real, Device, Index > :: getSerializationType()
{
   return HostType::getType();
};

template< typename Real,
          typename Device,
          typename Index >
String tnlMultiVector< 4, Real, Device, Index > :: getSerializationTypeVirtual() const
{
   return this->getSerializationType();
};

template< typename Real, typename Device, typename Index >
bool tnlMultiVector< 4, Real, Device, Index > :: setDimensions( const Index lSize,
                                                                       const Index kSize,
                                                                       const Index jSize,
                                                                       const Index iSize )
{
   Assert( iSize > 0 && jSize > 0 && kSize > 0 && lSize > 0,
              std::cerr << "iSize = " << iSize
                   << "jSize = " << jSize
                   << "kSize = " << kSize
                   << "lSize = " << lSize );

   dimensions[ 0 ] = iSize;
   dimensions[ 1 ] = jSize;
   dimensions[ 2 ] = kSize;
   dimensions[ 3 ] = lSize;
   return Vector< Real, Device, Index > :: setSize( iSize * jSize * kSize * lSize );
}

template< typename Real, typename Device, typename Index >
bool tnlMultiVector< 4, Real, Device, Index > :: setDimensions( const StaticVector< 4, Index >& dimensions )
{
   Assert( dimensions[ 0 ] > 0 && dimensions[ 1 ] > 0 && dimensions[ 2 ] && dimensions[ 3 ] > 0,
              std::cerr << "dimensions = " << dimensions );
   this->dimensions = dimensions;
   return Vector< Real, Device, Index > :: setSize( this->dimensions[ 3 ] *
                                                          this->dimensions[ 2 ] *
                                                          this->dimensions[ 1 ] *
                                                          this->dimensions[ 0 ] );
}

template< typename Real, typename Device, typename Index >
   template< typename MultiVector >
bool tnlMultiVector< 4, Real, Device, Index > :: setLike( const tnlMultiVector& multiVector )
{
   return setDimensions( multiVector. getDimensions() );
}

template< typename Real, typename Device, typename Index >
void tnlMultiVector< 4, Real, Device, Index > :: getDimensions( Index& lSize,
                                                                       Index& kSize,
                                                                       Index& jSize,
                                                                       Index& iSize ) const
{
   iSize = this->dimensions[ 0 ];
   jSize = this->dimensions[ 1 ];
   kSize = this->dimensions[ 2 ];
   lSize = this->dimensions[ 3 ];
}

template< typename Real, typename Device, typename Index >
const StaticVector< 4, Index >& tnlMultiVector< 4, Real, Device, Index > :: getDimensions() const
{
   return this->dimensions;
}

template< typename Real, typename Device, typename Index >
Index tnlMultiVector< 4, Real, Device, Index > :: getElementIndex( const Index l,
                                                                          const Index k,
                                                                          const Index j,
                                                                          const Index i ) const
{
   Assert( i >= 0 && i < this->dimensions[ 0 ] &&
              j >= 0 && j < this->dimensions[ 1 ] &&
              k >= 0 && k < this->dimensions[ 2 ] &&
              l >= 0 && l < this->dimensions[ 3 ],
              std::cerr << " i = " << i
                   << " j = " << j
                   << " k = " << k
                   << " l = " << l
                   << " this->dimensions = " << this->dimensions );
   return ( ( l * this->dimensions[ 2 ] + k ) * this->dimensions[ 1 ]  + j ) * this->dimensions[ 0 ] + i;
}

template< typename Real, typename Device, typename Index >
Real
tnlMultiVector< 4, Real, Device, Index >::
getElement( const Index l,
            const Index k,
            const Index j,
            const Index i ) const
{
   return Vector< Real, Device, Index > :: getElement( getElementIndex( l, k, j, i ) );
}

template< typename Real, typename Device, typename Index >
void tnlMultiVector< 4, Real, Device, Index > :: setElement( const Index l,
                                                                    const Index k,
                                                                    const Index j,
                                                                    const Index i, Real value )
{
   Vector< Real, Device, Index > :: setElement( getElementIndex( l, k, j, i ), value );
}


template< typename Real, typename Device, typename Index >
Real&
tnlMultiVector< 4, Real, Device, Index >::
operator()( const Index l,
            const Index k,
            const Index j,
            const Index i )
{
   return Vector< Real, Device, Index > :: operator[]( getElementIndex( l, k, j, i ) );
}

template< typename Real, typename Device, typename Index >
const Real& tnlMultiVector< 4, Real, Device, Index > :: operator()( const Index l,
                                                                               const Index k,
                                                                               const Index j,
                                                                               const Index i ) const
{
   return Vector< Real, Device, Index > :: operator[]( getElementIndex( l, k, j, i ) );
}

template< typename Real, typename Device, typename Index >
   template< typename MultiVector >
bool tnlMultiVector< 4, Real, Device, Index > :: operator == ( const MultiVector& vector ) const
{
   // TODO: Static assert on dimensions
   Assert( this->getDimensions() == vector. getDimensions(),
              std::cerr << "You are attempting to compare two Vectors with different dimensions." << std::endl
                   << "First vector dimensions are ( " << this->getDimensions() << " )" << std::endl
                   << "Second vector dimensions are ( " << vector. getDimensions() << " )" << std::endl; );
   return Vector< Real, Device, Index > :: operator == ( vector );
}

template< typename Real, typename Device, typename Index >
   template< typename MultiVector >
bool tnlMultiVector< 4, Real, Device, Index > :: operator != ( const MultiVector& vector ) const
{
   return ! ( (* this ) == vector );
}

template< typename Real, typename Device, typename Index >
tnlMultiVector< 4, Real, Device, Index >&
   tnlMultiVector< 4, Real, Device, Index > :: operator = ( const tnlMultiVector< 4, Real, Device, Index >& vector )
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
   template< typename MultiVector >
tnlMultiVector< 4, Real, Device, Index >&
   tnlMultiVector< 4, Real, Device, Index > :: operator = ( const MultiVector& vector )
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
bool tnlMultiVector< 4, Real, Device, Index > :: save( File& file ) const
{
   if( ! Vector< Real, Device, Index > :: save( file ) )
   {
      std::cerr << "I was not able to write the Vector of tnlMultiVector." << std::endl;
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
bool tnlMultiVector< 4, Real, Device, Index > :: load( File& file )
{
   if( ! Vector< Real, Device, Index > :: load( file ) )
   {
      std::cerr << "I was not able to read the Vector of tnlMultiVector." << std::endl;
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
std::ostream& operator << ( std::ostream& str, const tnlMultiVector< 4, Real, Device, Index >& Vector )
{
   for( Index l = 0; l < Vector. getDimensions()[ 3 ]; l ++ )
   {
      for( Index k = 0; k < Vector. getDimensions()[ 2 ]; k ++ )
      {
         for( Index j = 0; j < Vector. getDimensions()[ 1 ]; j ++ )
         {
            for( Index i = 0; i < Vector. getDimensions()[ 0 ]; i ++ )
            {
               str << Vector. getElement( l, k, j, i ) << " ";
            }
            str << std::endl;
         }
         str << std::endl;
      }
      str << std::endl;
   }
   return str;
}

template< typename Real, typename Device, typename Index >
bool tnlMultiVector< 4, Real, Device, Index > :: save( const String& fileName ) const
{
   return Object :: save( fileName );
}

template< typename Real, typename Device, typename Index >
bool tnlMultiVector< 4, Real, Device, Index > :: load( const String& fileName )
{
   return Object :: load( fileName );
}

#ifdef TEMPLATE_EXPLICIT_INSTANTIATION

#ifdef INSTANTIATE_FLOAT
extern template class tnlMultiVector< 4, float,  tnlHost, int >;
#endif
extern template class tnlMultiVector< 4, double, tnlHost, int >;
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
extern template class tnlMultiVector< 4, float,  tnlHost, long int >;
#endif
extern template class tnlMultiVector< 4, double, tnlHost, long int >;
#endif

#ifdef HAVE_CUDA
/*#ifdef INSTANTIATE_FLOAT
extern template class tnlMultiVector< 4, float,  tnlCuda, int >;
#endif
extern template class tnlMultiVector< 4, double, tnlCuda, int >;
#ifdef INSTANTIATE_LONG_INT
#ifdef INSTANTIATE_FLOAT
extern template class tnlMultiVector< 4, float,  tnlCuda, long int >;
#endif
extern template class tnlMultiVector< 4, double, tnlCuda, long int >;
#endif*/
#endif

#endif

} // namespace Vectors
} // namespace TNL
