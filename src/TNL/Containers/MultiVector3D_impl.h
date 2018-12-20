/***************************************************************************
                          MultiVector3D_impl.h  -  description
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
MultiVector< 3, Real, Device, Index > :: MultiVector()
{
}

template< typename Real, typename Device, typename Index >
String MultiVector< 3, Real, Device, Index > :: getType()
{
   return String( "Containers::MultiVector< ") +
          convertToString( Dimension ) +
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
String MultiVector< 3, Real, Device, Index > :: getTypeVirtual() const
{
   return this->getType();
};

template< typename Real,
          typename Device,
          typename Index >
String MultiVector< 3, Real, Device, Index > :: getSerializationType()
{
   return HostType::getType();
};

template< typename Real,
          typename Device,
          typename Index >
String MultiVector< 3, Real, Device, Index > :: getSerializationTypeVirtual() const
{
   return this->getSerializationType();
};

template< typename Real, typename Device, typename Index >
void MultiVector< 3, Real, Device, Index > :: setDimensions( const Index kSize,
                                                             const Index jSize,
                                                             const Index iSize )
{
   TNL_ASSERT( iSize > 0 && jSize > 0 && kSize > 0,
              std::cerr << "iSize = " << iSize
                   << "jSize = " << jSize
                   << "kSize = " << kSize );

   dimensions[ 0 ] = iSize;
   dimensions[ 1 ] = jSize;
   dimensions[ 2 ] = kSize;
   return Vector< Real, Device, Index > :: setSize( iSize * jSize * kSize );
}

template< typename Real, typename Device, typename Index >
void MultiVector< 3, Real, Device, Index > :: setDimensions( const StaticVector< 3, Index >& dimensions )
{
   TNL_ASSERT( dimensions[ 0 ] > 0 && dimensions[ 1 ] > 0 && dimensions[ 2 ],
              std::cerr << "dimensions = " << dimensions );
   this->dimensions = dimensions;
   Vector< Real, Device, Index > :: setSize( this->dimensions[ 2 ] *
                                             this->dimensions[ 1 ] *
                                             this->dimensions[ 0 ] );
}

template< typename Real, typename Device, typename Index >
   template< typename MultiVectorT >
void MultiVector< 3, Real, Device, Index > :: setLike( const MultiVectorT& multiVector )
{
   setDimensions( multiVector. getDimensions() );
}

template< typename Real, typename Device, typename Index >
void MultiVector< 3, Real, Device, Index > :: getDimensions( Index& kSize,
                                                             Index& jSize,
                                                             Index& iSize ) const
{
   iSize = this->dimensions[ 0 ];
   jSize = this->dimensions[ 1 ];
   kSize = this->dimensions[ 2 ];
}

template< typename Real, typename Device, typename Index >
const StaticVector< 3, Index >& MultiVector< 3, Real, Device, Index > :: getDimensions() const
{
   return this->dimensions;
}

template< typename Real, typename Device, typename Index >
Index MultiVector< 3, Real, Device, Index > :: getElementIndex( const Index k,
                                                                     const Index j,
                                                                     const Index i ) const
{
   TNL_ASSERT( i >= 0 && i < this->dimensions[ 0 ] &&
              j >= 0 && j < this->dimensions[ 1 ] &&
              k >= 0 && k < this->dimensions[ 2 ],
              std::cerr << " i = " << i
                   << " j = " << j
                   << " k = " << k
                   << " this->dimensions = " << this->dimensions );
   return ( k * this->dimensions[ 1 ]  + j ) * this->dimensions[ 0 ] + i;
}

template< typename Real, typename Device, typename Index >
Real MultiVector< 3, Real, Device, Index > :: getElement( const Index k,
                                                                  const Index j,
                                                                  const Index i ) const
{
   return Vector< Real, Device, Index > :: getElement( getElementIndex( k, j, i ) );
}

template< typename Real, typename Device, typename Index >
void MultiVector< 3, Real, Device, Index > :: setElement( const Index k,
                                                                    const Index j,
                                                                    const Index i, Real value )
{
   Vector< Real, Device, Index > :: setElement( getElementIndex( k, j, i ), value );
}


template< typename Real, typename Device, typename Index >
Real& MultiVector< 3, Real, Device, Index > :: operator()( const Index k,
                                                                        const Index j,
                                                                        const Index i )
{
   return Vector< Real, Device, Index > :: operator[]( getElementIndex( k, j, i ) );
}

template< typename Real, typename Device, typename Index >
const Real& MultiVector< 3, Real, Device, Index > :: operator()( const Index k,
                                                                               const Index j,
                                                                               const Index i ) const
{
   return Vector< Real, Device, Index > :: operator[]( getElementIndex( k, j, i ) );
}

template< typename Real, typename Device, typename Index >
   template< typename MultiVectorT >
bool MultiVector< 3, Real, Device, Index > :: operator == ( const MultiVectorT& vector ) const
{
   // TODO: Static assert on dimensions
   TNL_ASSERT( this->getDimensions() == vector. getDimensions(),
              std::cerr << "You are attempting to compare two Vectors with different dimensions." << std::endl
                   << "First vector dimensions are ( " << this->getDimensions() << " )" << std::endl
                   << "Second vector dimensions are ( " << vector. getDimensions() << " )" << std::endl; );
   return Vector< Real, Device, Index > :: operator == ( vector );
}

template< typename Real, typename Device, typename Index >
   template< typename MultiVectorT >
bool MultiVector< 3, Real, Device, Index > :: operator != ( const MultiVectorT& vector ) const
{
   return ! ( (* this ) == vector );
}

template< typename Real, typename Device, typename Index >
MultiVector< 3, Real, Device, Index >&
   MultiVector< 3, Real, Device, Index > :: operator = ( const MultiVector< 3, Real, Device, Index >& vector )
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
MultiVector< 3, Real, Device, Index >&
   MultiVector< 3, Real, Device, Index > :: operator = ( const MultiVectorT& vector )
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
bool MultiVector< 3, Real, Device, Index > :: save( File& file ) const
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
bool MultiVector< 3, Real, Device, Index > :: load( File& file )
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
std::ostream& operator << ( std::ostream& str, const MultiVector< 3, Real, Device, Index >& Vector )
{
   for( Index k = 0; k < Vector. getDimensions()[ 2 ]; k ++ )
   {
      for( Index j = 0; j < Vector. getDimensions()[ 1 ]; j ++ )
      {
         for( Index i = 0; i < Vector. getDimensions()[ 0 ]; i ++ )
         {
            str << Vector. getElement( k, j, i ) << " ";
         }
         str << std::endl;
      }
      str << std::endl;
   }
   return str;
}

template< typename Real, typename Device, typename Index >
bool MultiVector< 3, Real, Device, Index > :: save( const String& fileName ) const
{
   return Object :: save( fileName );
}

template< typename Real, typename Device, typename Index >
bool MultiVector< 3, Real, Device, Index > :: load( const String& fileName )
{
   return Object :: load( fileName );
}

} // namespace Containers
} // namespace TNL
