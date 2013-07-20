/***************************************************************************
                          tnlGridOld1D_impl.h  -  description
                             -------------------
    begin                : Nov 22, 2012
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

#ifndef tnlGridOld1D_IMPL_H_
#define tnlGridOld1D_IMPL_H_

#ifdef HAVE_CUDA
template< int Dimensions, typename Real, typename Index >
__global__ void setConstantDirichletBC( const Index xSize,
                                        const Real bc,
                                        Real* u );

template< int Dimensions, typename Real, typename Index >
__global__ void setDirichletBC( const Index xSize,
                                const Real* bc,
                                Real* u );

template< int Dimensions, typename Real, typename Index >
__global__ void setConstantNeumannBC( const Index xSize,
                                      const Real hx,
                                      const Real bc,
                                      Real* u );

template< int Dimensions, typename Real, typename Index >
__global__ void setNeumannBC( const Index xSize,
                              const Real hx,
                              const Real* bc,
                              Real* u );
#endif

template< typename Real, typename Device, typename Index >
tnlGridOld< 1, Real, Device, Index > :: tnlGridOld()
{
}


template< typename Real, typename Device, typename Index >
tnlGridOld< 1, Real, Device, Index > :: tnlGridOld( const tnlString& name )
{
   this -> setName( name );
}

template< typename Real, typename Device, typename Index >
tnlGridOld< 1, Real, Device, Index > :: tnlGridOld( const tnlString& name,
                                              const tnlGridOld< 1, Real, tnlHost, Index >& grid )
{
   this -> setName( name );
   this -> setLike( grid );
}

template< typename Real, typename Device, typename Index >
tnlGridOld< 1, Real, Device, Index > :: tnlGridOld( const tnlString& name,
                                              const tnlGridOld< 1, Real, tnlCuda, Index >& grid )
{
   this -> setName( name );
   this -> setLike( grid );
}

template< typename Real, typename Device, typename Index >
const tnlTuple< 1, Index >& tnlGridOld< 1, Real, Device, Index > :: getDimensions() const
{
   return tnlMultiVector< 1, Real, Device, Index > :: getDimensions();
}

template< typename Real, typename Device, typename Index >
bool tnlGridOld< 1, Real, Device, Index > :: setDimensions( const tnlTuple< 1, Index >& dimensions )
{
   if( ! tnlMultiVector< 1, Real, Device, Index > :: setDimensions( dimensions ) )
      return false;
   spaceSteps[ 0 ] = ( domainUpperCorner[ 0 ] - domainLowerCorner[ 0 ] ) / ( Real ) ( this -> getDimensions()[ 0 ] - 1 );
   return true;
}

template< typename Real, typename Device, typename Index >
bool tnlGridOld< 1, Real, Device, Index > :: setDomain( const tnlTuple< 1, Real >& origin,
                                                     const tnlTuple< 1, Real >& proportions )
{
   if( origin >= proportions )
   {
      cerr << "Wrong parameters for the grid domain of " << this -> getName() << ". The low corner must by smaller than the high corner." << endl
               << "origin = " << origin << endl << "proportions = " << proportions << endl;
      return false;
   }
   domainLowerCorner = origin;
   domainUpperCorner = proportions;
   spaceSteps[ 0 ] = ( domainUpperCorner[ 0 ] - domainLowerCorner[ 0 ] ) / ( Real ) ( this -> getDimensions()[ 0 ] - 1 );
   return true;
}

template< typename Real, typename Device, typename Index >
   template< typename Grid >
bool tnlGridOld< 1, Real, Device, Index > :: setLike( const Grid& v )
{
   return tnlMultiVector< 1, Real, Device, Index > :: setDimensions( v. getDimensions() ) &&
          this -> setDomain( v. getDomainLowerCorner(), v. getDomainUpperCorner() );
}

template< typename Real, typename Device, typename Index >
const tnlTuple< 1, Real >& tnlGridOld< 1, Real, Device, Index > :: getDomainLowerCorner() const
{
   return this -> domainLowerCorner;
}

template< typename Real, typename Device, typename Index >
const tnlTuple< 1, Real >& tnlGridOld< 1, Real, Device, Index > :: getDomainUpperCorner() const
{
   return this -> domainUpperCorner;
}

template< typename Real, typename Device, typename Index >
const tnlTuple< 1, Real >& tnlGridOld< 1, Real, Device, Index > :: getSpaceSteps() const
{
   return spaceSteps;
}

template< typename Real, typename Device, typename Index >
tnlString tnlGridOld< 1, Real, Device, Index > :: getType() const
{
   return tnlString( "tnlGridOld< ") +
          tnlString( "1" ) +
          tnlString( ", " ) +
          tnlString( getParameterType< Real >() ) +
          tnlString( ", " ) +
          Device :: getDeviceType() +
          tnlString( ", " ) +
          tnlString( getParameterType< Index >() ) +
          tnlString( " >" );
}

template< typename Real, typename Device, typename Index >
bool tnlGridOld< 1, Real, Device, Index > :: operator == ( const tnlGridOld< 1, Real, Device, Index >& grid ) const
{
   tnlAssert( this -> getDomainLowerCorner() == grid. getDomainLowerCorner() &&
              this -> getDomainUpperCorner() == grid. getDomainUpperCorner(),
              cerr << "You are attempting to compare two grids with different domains." << endl
                   << "First grid name is " << this -> getName()
                   << " domain is ( " << this -> getDomainLowerCorner() << " )- ("
                                      << this -> getDomainUpperCorner() << ")" << endl
                   << "Second grid is " << grid. getName()
                   << " domain is ( " << grid. getDomainLowerCorner() << " )- ("
                                      << grid. getDomainUpperCorner() << ")" << endl; );
   return tnlMultiVector< 1, Real, Device, Index > :: operator == ( grid );
}

template< typename Real, typename Device, typename Index >
bool tnlGridOld< 1, Real, Device, Index > :: operator != ( const tnlGridOld< 1, Real, Device, Index >& grid ) const
{
   return ! ( (* this ) == grid );
}

template< typename Real, typename Device, typename Index >
  template< typename Real2, typename Device2, typename Index2 >
tnlGridOld< 1, Real, Device, Index >&
tnlGridOld< 1, Real, Device, Index >
:: operator = ( const tnlGridOld< 1, Real2, Device2, Index2 >& grid )
{
   tnlMultiVector< 1, Real, Device, Index > :: operator = ( grid );
   return ( *this );
}

template< typename Real, typename Device, typename Index >
Real tnlGridOld< 1, Real, Device, Index > :: getValue( const tnlTuple< 1, Real >& point ) const
{
   tnlAssert( 0, cerr << "Interpolation is not implemented yet.");
}

template< typename Real, typename Device, typename Index >
Real tnlGridOld< 1, Real, Device, Index > :: getValue( const Real& x ) const
{
   return this -> getValue( tnlTuple< 1, Real >( x ) );
}

template< typename Real, typename Device, typename Index >
Real tnlGridOld< 1, Real, Device, Index > :: Partial_x_f( const Index i1 ) const
{
   tnlAssert( i1 >= 0 &&
              i1 < ( this -> getDimensions(). x() - 1 ),
              cerr << " i1 = " << i1 << " and it should be in [ 0, " <<
                   ( this -> getDimensions(). y() - 1  ) << " )" << endl; );

   const Real& Hx = spaceSteps[ 0 ];
   tnlAssert( Hx > 0, cerr << "Hx = " << Hx << endl; );
   return ( this -> getElementIndex( i1 + 1 ) -
            this ->  getElementIndex( i1 ) ) / Hx;
};

template< typename Real, typename Device, typename Index >
Real tnlGridOld< 1, Real, Device, Index > :: Partial_x_b( const Index i1 ) const
{
   tnlAssert( i1 > 0 &&
              i1 <= ( this -> getDimensions()[ tnlX ] - 1 ),
              cerr << " i1 = " << i1 << " and it should be in ( 0, " <<
                   ( this -> getDimensions()[ tnlX ] - 1  ) << " ] " << endl; );

   const Real& Hx = spaceSteps[ 0 ];
   tnlAssert( Hx > 0, cerr << "Hx = " << Hx << endl; );
   return ( this -> getElement( i1 ) -
            this -> getElement( i1 - 1 ) ) / Hx;
};

template< typename Real, typename Device, typename Index >
Real tnlGridOld< 1, Real, Device, Index > :: Partial_x( const Index i1 ) const
{
   tnlAssert( i1 > 0 &&
              i1 < ( this -> getDimensions()[ tnlX ] - 1 ),
              cerr << " i1 = " << i1 << " and it should be in ( 0, " <<
                   ( this -> getDimensions()[ tnlX ] - 1  ) << " ) " << endl; );

   const Real& Hx = spaceSteps[ 0 ];
   tnlAssert( Hx > 0, cerr << "Hx = " << Hx << endl; );
   return ( this -> getElement( i1 + 1 ) -
            this -> getElement( i1 - 1 ) ) / ( 2.0 * Hx );
};

template< typename Real, typename Device, typename Index >
Real tnlGridOld< 1, Real, Device, Index > :: Partial_xx( const Index i1 ) const
{
   tnlAssert( i1 > 0 &&
              i1 < ( this -> getDimensions()[ tnlX ] - 1 ),
              cerr << " i1 = " << i1 << " and it should be in ( 0, " <<
                   ( this -> getDimensions()[ tnlX ] - 1  ) << " ) " << endl; );

   const Real& Hx = spaceSteps[ 0 ];
   tnlAssert( Hx > 0, cerr << "Hx = " << Hx << endl; );
   return ( this -> getElement( i1 + 1 ) -
            2.0 * this -> getElement( i1 ) +
            this -> getElement( i1 - 1 ) ) / ( Hx * Hx );
};


template< typename Real, typename Device, typename Index >
void tnlGridOld< 1, Real, Device, Index > :: setDirichletBC( const tnlGridOld< 1, Real, Device, Index >&bc,
                                                          const tnlTuple< 1, bool >& lowerBC,
                                                          const tnlTuple< 1, bool >& upperBC )
{
   if( Device :: getDevice() == tnlHostDevice )
   {
      const Index xSize = this -> getDimensions(). x();
      if( lowerBC. x() )
         ( *this )( 0 ) = bc( 0 );
      if( upperBC. x() )
         ( *this )( xSize - 1 ) = bc( xSize - 1 );
   }
   if( Device :: getDevice() == tnlCudaDevice )
   {
#ifdef HAVE_CUDA
      {
         tnlAssert( false, );
         /*const Index size = this -> getSize();
         const Index desBlockSize = 64;
         const Index gridSize = size / desBlockSize + ( size % desBlockSize != 0 );
         dim3 gridDim( 0 ), blockDim( 0 );
         gridDim. x = gridSize;
         blockDim. x = desBlockSize;
         :: setDirichletBC< 2, Real, Index ><<< gridDim, blockDim >>>( this -> getDimensions(). x(),
                                                                       this -> getDimensions(). y(),
                                                                       bc. getData(),
                                                                       this -> getData() );*/
      }
#endif
   }
}

template< typename Real, typename Device, typename Index >
void tnlGridOld< 1, Real, Device, Index > :: setDirichletBC( const Real& bcValue,
                                                                   const tnlTuple< 1, bool >& lowerBC,
                                                                   const tnlTuple< 1, bool >& upperBC )
{
   if( Device :: getDevice() == tnlHostDevice )
   {
      const Index xSize = this -> getDimensions(). x();
      if( lowerBC. x() )
         ( *this )( 0 ) = bcValue;
      if( upperBC. x() )
         ( *this )( xSize - 1 ) = bcValue;
   }
   if( Device :: getDevice() == tnlCudaDevice )
   {
#ifdef HAVE_CUDA
      tnlAssert( false, );
      /*{
         const Index size = this -> getSize();
         const Index desBlockSize = 64;
         const Index gridSize = size / desBlockSize + ( size % desBlockSize != 0 );
         dim3 gridDim( 0 ), blockDim( 0 );
         gridDim. x = gridSize;
         blockDim. x = desBlockSize;
         :: setConstantDirichletBC< 2, Real, Index ><<< gridDim, blockDim >>>( this -> getDimensions(). x(),
                                                                               this -> getDimensions(). y(),
                                                                               bcValue,
                                                                               this -> getData() );
      }*/
#endif
   }
}

template< typename Real, typename Device, typename Index >
void tnlGridOld< 1, Real, Device, Index > :: setNeumannBC( const tnlGridOld< 1, Real, Device, Index >&bc,
                                                        const tnlTuple< 1, bool >& lowerBC,
                                                        const tnlTuple< 1, bool >& upperBC )
{
   if( Device :: getDevice() == tnlHostDevice )
   {
      const Real& hx = this -> getSpaceSteps(). x();

      const Index xSize = this -> getDimensions(). x();
      if( lowerBC. x() )
         ( *this )( 0 ) = ( *this )( 1 ) + hx * bc( 0 );
      if( upperBC. x() )
         ( *this )( xSize - 1 ) = ( *this )( xSize - 1 ) + hx * bc( xSize -1 );
   }
   if( Device :: getDevice() == tnlCudaDevice )
   {
#ifdef HAVE_CUDA
      tnlAssert( false, );
      /*{
         const Index size = this -> getSize();
         const Index desBlockSize = 64;
         const Index gridSize = size / desBlockSize + ( size % desBlockSize != 0 );
         dim3 gridDim( 0 ), blockDim( 0 );
         gridDim. x = gridSize;
         blockDim. x = desBlockSize;
         ::setNeumannBC< 2, Real, Index ><<< gridDim, blockDim >>>( this -> getDimensions(). x(),
                                                                    this -> getDimensions(). y(),
                                                                    this -> getSpaceSteps(). x(),
                                                                    this -> getSpaceSteps(). y(),
                                                                    bc. getData(),
                                                                    this -> getData() );
      }*/
#endif
   }
}

template< typename Real, typename Device, typename Index >
void tnlGridOld< 1, Real, Device, Index > :: setNeumannBC( const Real& bcValue,
                                                                 const tnlTuple< 1, bool >& lowerBC,
                                                                 const tnlTuple< 1, bool >& upperBC )
{
   if( Device :: getDevice() == tnlHostDevice )
   {
      const Real& hx = this -> getSpaceSteps(). x();

      const Index xSize = this -> getDimensions(). x();
      if( lowerBC. x() )
         ( *this )( 0 ) = ( *this )( 1 ) + hx * bcValue;
      if( upperBC. x() )
         ( *this )( xSize - 1 ) = ( *this )( xSize - 1 ) + hx * bcValue;
   }
   if( Device :: getDevice() == tnlCudaDevice )
   {
#ifdef HAVE_CUDA
      tnlAssert( false, )
      /*{
         const Index size = this -> getSize();
         const Index desBlockSize = 64;
         const Index gridSize = size / desBlockSize + ( size % desBlockSize != 0 );
         dim3 gridDim( 0 ), blockDim( 0 );
         gridDim. x = gridSize;
         blockDim. x = desBlockSize;
         :: setConstantNeumannBC< 2, Real, Index ><<< gridDim, blockDim >>>( this -> getDimensions(). x(),
                                                                             this -> getDimensions(). y(),
                                                                             this -> getSpaceSteps(). x(),
                                                                             this -> getSpaceSteps(). y(),
                                                                             bcValue,
                                                                             this -> getData() );
      }*/
#endif
   }
}

/*
template< typename Real, typename Device, typename Index >
Real tnlGridOld< 1, Real, Device, Index > :: getMax() const
{
   return tnlMax( * this );
}

template< typename Real, typename Device, typename Index >
Real tnlGridOld< 1, Real, Device, Index > :: getMin() const
{
   return tnlMin( * this );
}

template< typename Real, typename Device, typename Index >
Real tnlGridOld< 1, Real, Device, Index > :: getAbsMax() const
{
   return tnlAbsMax( * this );
}

template< typename Real, typename Device, typename Index >
Real tnlGridOld< 1, Real, Device, Index > :: getAbsMin() const
{
   return tnlAbsMin( * this );
}
*/

template< typename Real, typename Device, typename Index >
Real tnlGridOld< 1, Real, Device, Index > :: getLpNorm( const Real& p ) const
{
   Real result = this -> lpNorm( p );
   return result * getSpaceSteps(). x();
}

/*template< typename Real, typename Device, typename Index >
Real tnlGridOld< 1, Real, Device, Index > :: getSum() const
{
   return tnlMultiVector< 1, Real, Device, Index > :: sum( * this );
}

template< typename Real, typename Device, typename Index >
Real tnlGridOld< 1, Real, Device, Index > :: getDifferenceMax( const tnlVector< Real, Device, Index >& v ) const
{
   tnlAssert( this -> getDimensions() == v. getDimensions(),
              cerr << "The grid names are " << this -> getName()
                   << " and " << v. getName()
                   << "To get grids with the same parameters use the method setLike." << endl; );
   tnlAssert( this -> getDomainLoweCorner() == v. getDomainLowerCorner() &&
              this -> getDomainUpperCorner() == v. getDomainUpperCorner(),
              cerr << "The grid names are " << this -> getName()
                   << " and " << v. getName()
                   << "To get grids with the same parameters use the method setLike." << endl; );
   return v. differenceMax( *this );
}

template< typename Real, typename Device, typename Index >
Real tnlGridOld< 1, Real, Device, Index > :: getDifferenceMin( const tnlVector< Real, Device, Index >& v ) const
{
   tnlAssert( this -> getDimensions() == v. getDimensions(),
              cerr << "The grid names are " << this -> getName()
                   << " and " << v. getName()
                   << "To get grids with the same parameters use the method setLike." << endl; );
   tnlAssert( this -> getDomainLoweCorner() == v. getDomainLowerCorner() &&
              this -> getDomainUpperCorner() == v. getDomainUpperCorner(),
              cerr << "The grid names are " << this -> getName()
                   << " and " << v. getName()
                   << "To get grids with the same parameters use the method setLike." << endl; );
   return v. differenceMin( *this );
}

template< typename Real, typename Device, typename Index >
Real tnlGridOld< 1, Real, Device, Index > :: getDifferenceAbsMax( const tnlVector< Real, Device, Index >& v ) const
{
   tnlAssert( this -> getDimensions() == v. getDimensions(),
              cerr << "The grid names are " << this -> getName()
                   << " and " << v. getName()
                   << "To get grids with the same parameters use the method setLike." << endl; );
   tnlAssert( this -> getDomainLoweCorner() == v. getDomainLowerCorner() &&
              this -> getDomainUpperCorner() == v. getDomainUpperCorner(),
              cerr << "The grid names are " << this -> getName()
                   << " and " << v. getName()
                   << "To get grids with the same parameters use the method setLike." << endl; );
   return v. differenceAbsMax( *this );
}

template< typename Real, typename Device, typename Index >
Real tnlGridOld< 1, Real, Device, Index > :: getDifferenceAbsMin( const tnlVector< Real, Device, Index >& v ) const
{
   tnlAssert( this -> getDimensions() == v. getDimensions(),
              cerr << "The grid names are " << this -> getName()
                   << " and " << v. getName()
                   << "To get grids with the same parameters use the method setLike." << endl; );
   tnlAssert( this -> getDomainLoweCorner() == v. getDomainLowerCorner() &&
              this -> getDomainUpperCorner() == v. getDomainUpperCorner(),
              cerr << "The grid names are " << this -> getName()
                   << " and " << v. getName()
                   << "To get grids with the same parameters use the method setLike." << endl; );
   return v. differenceAbsMin( *this );
}
*/

template< typename Real, typename Device, typename Index >
Real tnlGridOld< 1, Real, Device, Index > :: getDifferenceLpNorm( const tnlVector< Real, Device, Index >& v, const Real& p ) const
{
   tnlAssert( this -> getDimensions() == v. getDimensions(),
              cerr << "The grid names are " << this -> getName()
                   << " and " << v. getName()
                   << "To get grids with the same parameters use the method setLike." << endl; );
   tnlAssert( this -> getDomainLoweCorner() == v. getDomainLowerCorner() &&
              this -> getDomainUpperCorner() == v. getDomainUpperCorner(),
              cerr << "The grid names are " << this -> getName()
                   << " and " << v. getName()
                   << "To get grids with the same parameters use the method setLike." << endl; );

   Real result = v. differenceLpNorm( * this, p );
   return result * getSpaceSteps(). x();
}

/*
template< typename Real, typename Device, typename Index >
Real tnlGridOld< 1, Real, Device, Index > :: getDifferenceSum( const tnlVector< Real, Device, Index >& v ) const
{
   tnlAssert( this -> getDimensions() == v. getDimensions(),
              cerr << "The grid names are " << this -> getName()
                   << " and " << v. getName()
                   << "To get grids with the same parameters use the method setLike." << endl; );
   tnlAssert( this -> getDomainLoweCorner() == v. getDomainLowerCorner() &&
              this -> getDomainUpperCorner() == v. getDomainUpperCorner(),
              cerr << "The grid names are " << this -> getName()
                   << " and " << v. getName()
                   << "To get grids with the same parameters use the method setLike." << endl; );
   return this -> differenceSum( v );
}

template< typename Real, typename Device, typename Index >
void tnlGridOld< 1, Real, Device, Index > :: scalarMultiplication( const Real& alpha )
{
   return this -> scalarMultiplication( alpha );
}

template< typename Real, typename Device, typename Index >
Real tnlGridOld< 1, Real, Device, Index > :: scalarProduct( const tnlVector< Real, Device, Index >& v ) const
{
   tnlAssert( this -> getDimensions() == v. getDimensions(),
              cerr << "The grid names are " << this -> getName()
                   << " and " << v. getName()
                   << "To get grids with the same parameters use the method setLike." << endl; );
   tnlAssert( this -> getDomainLoweCorner() == v. getDomainLowerCorner() &&
              this -> getDomainUpperCorner() == v. getDomainUpperCorner(),
              cerr << "The grid names are " << this -> getName()
                   << " and " << v. getName()
                   << "To get grids with the same parameters use the method setLike." << endl; );
   return this -> scalarProduct( v );
};

template< typename Real, typename Device, typename Index >
void tnlGridOld< 1, Real, Device, Index > :: saxpy( const Real& alpha,
                                                         const tnlVector< Real, Device, Index >& x )
{
   tnlAssert( this -> getDimensions() == v. getDimensions(),
              cerr << "The grid names are " << this -> getName()
                   << " and " << v. getName()
                   << "To get grids with the same parameters use the method setLike." << endl; );
   tnlAssert( this -> getDomainLoweCorner() == v. getDomainLowerCorner() &&
              this -> getDomainUpperCorner() == v. getDomainUpperCorner(),
              cerr << "The grid names are " << this -> getName()
                   << " and " << v. getName()
                   << "To get grids with the same parameters use the method setLike." << endl; );
   this -> saxpy( alpha, x );
};

template< typename Real, typename Device, typename Index >
void tnlGridOld< 1, Real, Device, Index > :: saxmy( const Real& alpha,
                                                     const tnlVector< Real, Device, Index >& x )
{
   tnlAssert( this -> getDimensions() == v. getDimensions(),
              cerr << "The grid names are " << this -> getName()
                   << " and " << v. getName()
                   << "To get grids with the same parameters use the method setLike." << endl; );
   tnlAssert( this -> getDomainLoweCorner() == v. getDomainLowerCorner() &&
              this -> getDomainUpperCorner() == v. getDomainUpperCorner(),
              cerr << "The grid names are " << this -> getName()
                   << " and " << v. getName()
                   << "To get grids with the same parameters use the method setLike." << endl; );
   this -> saxmy( alpha, x );
};
*/

template< typename Real, typename Device, typename Index >
bool tnlGridOld< 1, Real, Device, Index > :: save( tnlFile& file ) const
{
   if( ! tnlMultiVector< 1, Real, Device, Index > :: save( file ) )
   {
      cerr << "I was not able to write the tnlMultiVector of the tnlGridOld "
           << this -> getName() << endl;
      return false;
   }
   if( ! domainLowerCorner. save( file ) ||
       ! domainUpperCorner. save( file ) ||
       ! spaceSteps. save( file ) )
   {
      cerr << "I was not able to write the domain decsription of the tnlGridOld "
           << this -> getName() << endl;
      return false;
   }
   return true;
}

template< typename Real, typename Device, typename Index >
bool tnlGridOld< 1, Real, Device, Index > :: load( tnlFile& file )
{
   if( ! tnlMultiVector< 1, Real, Device, Index > :: load( file ) )
   {
      cerr << "I was not able to read the tnlMultiVector of the tnlGridOld "
           << this -> getName() << endl;
      return false;
   }
   if( ! domainLowerCorner. load( file ) ||
       ! domainUpperCorner. load( file ) ||
       ! spaceSteps. load( file ) )
   {
      cerr << "I was not able to read the domain description of the tnlGridOld "
           << this -> getName() << endl;
      return false;
   }
   return true;
}

template< typename Real, typename Device, typename Index >
bool tnlGridOld< 1, Real, Device, Index > :: save( const tnlString& fileName ) const
{
   return tnlObject :: save( fileName );
}

template< typename Real, typename Device, typename Index >
bool tnlGridOld< 1, Real, Device, Index > :: load( const tnlString& fileName )
{
   return tnlObject :: load( fileName );
}

template< typename Real, typename Device, typename Index >
bool tnlGridOld< 1, Real, Device, Index > :: draw( const tnlString& fileName,
                                                         const tnlString& format,
                                                         const tnlTuple< 1, Index > steps ) const
{
   tnlAssert( steps > ( tnlTuple< 1, Index >( 0 ) ),
              cerr << "Wrong steps of increment ( " << steps << " )"
                   << " for drawing the tnlGridOld " << this -> getName() << "." << endl; );
   if( format == "tnl" )
      return this -> save( fileName );
   fstream file;
   file. open( fileName. getString(), ios :: out );
   if( ! file )
   {
      cerr << " I am not able to open the file " << fileName
           << " for drawing the tnlGridOld " << this -> getName() << "." << endl;
      return false;
   }
   if( format == "gnuplot" )
   {
      const Index xSize = this -> getDimensions()[ tnlX ];
      const Real& ax = this -> getDomainLowerCorner()[ tnlX ];
      const Real& hx = this -> getSpaceSteps()[ tnlX ];
      for( Index i = 0; i < xSize; i += steps[ tnlX ] )
               file << setprecision( 12 )
                    << ax + i * hx * steps[ tnlX ]
                    << " "
                    << this -> getElement( i )
                    << endl;
      return true;
   }
   cerr << endl << "I do not know a format " << format << " for tnlGridOld with 2 dimensions.";
   return false;
}

#ifdef HAVE_CUDA
template< int Dimensions, typename Real, typename Index >
__global__ void  setConstantDirichletBC( const Index xSize,
                                         const Real bc,
                                         Real* u )
{
   /*const Index ij = blockIdx. x * blockDim. x + threadIdx. x;
   const Index i = ij / ySize;
   const Index j = ij % ySize;

   if( ij < xSize * ySize &&
       ( i == 0 || j == 0 || i == xSize - 1 || j == ySize - 1 ) )
   {
      u[ ij ] = bc;
   }*/
}

template< int Dimensions, typename Real, typename Index >
__global__ void setDirichletBC( const Index xSize,
                                const Real* bc,
                                Real* u )
{
   /*const Index ij = blockIdx. x * blockDim. x + threadIdx. x;
   const Index i = ij / ySize;
   const Index j = ij % ySize;

   if( ij < xSize * ySize &&
       ( i == 0 || j == 0 || i == xSize - 1 || j == ySize - 1 ) )
   {
      u[ ij ] = bc[ ij ];
   }*/
}

template< int Dimensions, typename Real, typename Index >
__global__ void setConstantNeumannBC( const Index xSize,
                                      const Real hx,
                                      const Real bc,
                                      Real* u )
{
   /*const Index ij = blockIdx. x * blockDim. x + threadIdx. x;
   const Index i = ij / ySize;
   const Index j = ij % ySize;

   if( ij < xSize * ySize &&
       ( i == 0 || j == 0 || i == xSize - 1 || j == ySize - 1 ) )
   {
     if( i == 0 )
        u[ ij ] = u[ ij + ySize ] + hx * bc;
     if( i == xSize - 1 )
        u[ ij ] = u[ ij - ySize ] + hx * bc;

     __syncthreads();

     if( j == 0 )
        u[ ij ] = u[ ij + 1 ] + hy * bc;
     if( j == ySize - 1 )
        u[ ij ] = u[ ij - 1 ] + hy * bc;
   }*/
}

template< int Dimensions, typename Real, typename Index >
__global__ void setNeumannBC( const Index xSize,
                              const Real hx,
                              const Real* bc,
                              Real* u )
{
   /*const Index ij = blockIdx. x * blockDim. x + threadIdx. x;
   const Index i = ij / ySize;
   const Index j = ij % ySize;

   if( ij < xSize * ySize &&
       ( i == 0 || j == 0 || i == xSize - 1 || j == ySize - 1 ) )
   {
     if( i == 0 )
        u[ ij ] = u[ ij + ySize ] + hx * bc[ ij ];
     if( i == xSize - 1 )
        u[ ij ] = u[ ij - ySize ] + hx * bc[ ij ];

     __syncthreads();

     if( j == 0 )
        u[ ij ] = u[ ij + 1 ] + hy * bc[ ij ];
     if( j == ySize - 1 )
        u[ ij ] = u[ ij - 1 ] + hy * bc[ ij ];
   }*/
}

#endif




#endif /* tnlGrid1D_IMPL_H_ */
