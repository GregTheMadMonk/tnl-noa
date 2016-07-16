/***************************************************************************
                          tnlMultiArray4D_impl.h  -  description
                             -------------------
    begin                : Nov 13, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLMULTIARRAY4D_IMPL_H_
#define TNLMULTIARRAY4D_IMPL_H_



template< typename Element, typename Device, typename Index >
tnlMultiArray< 4, Element, Device, Index > :: tnlMultiArray()
{
}

template< typename Element, typename Device, typename Index >
tnlString tnlMultiArray< 4, Element, Device, Index > :: getType()
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
tnlString tnlMultiArray< 4, Element, Device, Index > :: getTypeVirtual() const
{
   return this->getType();
};

template< typename Element,
          typename Device,
          typename Index >
tnlString tnlMultiArray< 4, Element, Device, Index > :: getSerializationType()
{
   return HostType::getType();
};

template< typename Element,
          typename Device,
          typename Index >
tnlString tnlMultiArray< 4, Element, Device, Index > :: getSerializationTypeVirtual() const
{
   return this->getSerializationType();
};

template< typename Element, typename Device, typename Index >
bool tnlMultiArray< 4, Element, Device, Index > :: setDimensions( const Index lSize,
                                                                       const Index kSize,
                                                                       const Index jSize,
                                                                       const Index iSize )
{
   tnlAssert( iSize > 0 && jSize > 0 && kSize > 0 && lSize > 0,
              cerr << "iSize = " << iSize
                   << "jSize = " << jSize
                   << "kSize = " << kSize
                   << "lSize = " << lSize );

   dimensions[ 0 ] = iSize;
   dimensions[ 1 ] = jSize;
   dimensions[ 2 ] = kSize;
   dimensions[ 3 ] = lSize;
   return tnlArray< Element, Device, Index > :: setSize( iSize * jSize * kSize * lSize );
}

template< typename Element, typename Device, typename Index >
bool tnlMultiArray< 4, Element, Device, Index > :: setDimensions( const tnlStaticVector< 4, Index >& dimensions )
{
   tnlAssert( dimensions[ 0 ] > 0 && dimensions[ 1 ] > 0 && dimensions[ 2 ] && dimensions[ 3 ] > 0,
              cerr << "dimensions = " << dimensions );
   /****
    * Swap the dimensions in the tuple to be compatible with the previous method.
    */
   this->dimensions[ 0 ] = dimensions[ 3 ];
   this->dimensions[ 1 ] = dimensions[ 2 ];
   this->dimensions[ 2 ] = dimensions[ 1 ];
   this->dimensions[ 3 ] = dimensions[ 0 ];
   return tnlArray< Element, Device, Index > :: setSize( this->dimensions[ 3 ] *
                                                         this->dimensions[ 2 ] *
                                                         this->dimensions[ 1 ] *
                                                         this->dimensions[ 0 ] );
}

template< typename Element, typename Device, typename Index >
   template< typename MultiArray >
bool tnlMultiArray< 4, Element, Device, Index > :: setLike( const MultiArray& multiArray )
{
   return setDimensions( multiArray. getDimensions() );
}

template< typename Element, typename Device, typename Index >
void tnlMultiArray< 4, Element, Device, Index >::reset()
{
   this->dimensions = tnlStaticVector< 4, Index >( ( Index ) 0 );
   tnlArray< Element, Device, Index >::reset();
}

template< typename Element, typename Device, typename Index >
__cuda_callable__
void tnlMultiArray< 4, Element, Device, Index > :: getDimensions( Index& lSize,
                                                                       Index& kSize,
                                                                       Index& jSize,
                                                                       Index& iSize ) const
{
   iSize = this->dimensions[ 0 ];
   jSize = this->dimensions[ 1 ];
   kSize = this->dimensions[ 2 ];
   lSize = this->dimensions[ 3 ];
}

template< typename Element, typename Device, typename Index >
__cuda_callable__
const tnlStaticVector< 4, Index >& tnlMultiArray< 4, Element, Device, Index > :: getDimensions() const
{
   return this->dimensions;
}

template< typename Element, typename Device, typename Index >
__cuda_callable__
Index tnlMultiArray< 4, Element, Device, Index > :: getElementIndex( const Index l,
                                                                     const Index k,
                                                                     const Index j,
                                                                     const Index i ) const
{
   tnlAssert( i >= 0 && i < this->dimensions[ 0 ] &&
              j >= 0 && j < this->dimensions[ 1 ] &&
              k >= 0 && k < this->dimensions[ 2 ] &&
              l >= 0 && l < this->dimensions[ 3 ],
              cerr << " i = " << i
                   << " j = " << j
                   << " k = " << k
                   << " l = " << l
                   << " this->dimensions = " << this->dimensions );
   return ( ( l * this->dimensions[ 2 ] + k ) * this->dimensions[ 1 ]  + j ) * this->dimensions[ 0 ] + i;
}

template< typename Element, typename Device, typename Index >
Element tnlMultiArray< 4, Element, Device, Index > :: getElement( const Index l,
                                                                       const Index k,
                                                                       const Index j,
                                                                       const Index i ) const
{
   return tnlArray< Element, Device, Index > :: getElement( getElementIndex( l, k, j, i ) );
}

template< typename Element, typename Device, typename Index >
void tnlMultiArray< 4, Element, Device, Index > :: setElement( const Index l,
                                                                    const Index k,
                                                                    const Index j,
                                                                    const Index i, Element value )
{
   tnlArray< Element, Device, Index > :: setElement( getElementIndex( l, k, j, i ), value );
}


template< typename Element, typename Device, typename Index >
__cuda_callable__
Element& tnlMultiArray< 4, Element, Device, Index > :: operator()( const Index l,
                                                                        const Index k,
                                                                        const Index j,
                                                                        const Index i )
{
   return tnlArray< Element, Device, Index > :: operator[]( getElementIndex( l, k, j, i ) );
}

template< typename Element, typename Device, typename Index >
__cuda_callable__
const Element& tnlMultiArray< 4, Element, Device, Index > :: operator()( const Index l,
                                                                               const Index k,
                                                                               const Index j,
                                                                               const Index i ) const
{
   return tnlArray< Element, Device, Index > :: operator[]( getElementIndex( l, k, j, i ) );
}

template< typename Element, typename Device, typename Index >
   template< typename MultiArray >
bool tnlMultiArray< 4, Element, Device, Index > :: operator == ( const MultiArray& array ) const
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
bool tnlMultiArray< 4, Element, Device, Index > :: operator != ( const MultiArray& array ) const
{
   return ! ( (* this ) == array );
}

template< typename Element, typename Device, typename Index >
tnlMultiArray< 4, Element, Device, Index >&
   tnlMultiArray< 4, Element, Device, Index > :: operator = ( const tnlMultiArray< 4, Element, Device, Index >& array )
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
tnlMultiArray< 4, Element, Device, Index >&
   tnlMultiArray< 4, Element, Device, Index > :: operator = ( const MultiArray& array )
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
bool tnlMultiArray< 4, Element, Device, Index > :: save( tnlFile& file ) const
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
bool tnlMultiArray< 4, Element, Device, Index > :: load( tnlFile& file )
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
ostream& operator << ( ostream& str, const tnlMultiArray< 4, Element, Device, Index >& array )
{
   for( Index l = 0; l < array. getDimensions()[ 3 ]; l ++ )
   {
      for( Index k = 0; k < array. getDimensions()[ 2 ]; k ++ )
      {
         for( Index j = 0; j < array. getDimensions()[ 1 ]; j ++ )
         {
            for( Index i = 0; i < array. getDimensions()[ 0 ]; i ++ )
            {
               str << array. getElement( l, k, j, i ) << " ";
            }
            str << endl;
         }
         str << endl;
      }
      str << endl;
   }
   return str;
}

#endif /* TNLMULTIARRAY4D_IMPL_H_ */
