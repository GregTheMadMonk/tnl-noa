/***************************************************************************
                          tnlGridOld3D_impl.h  -  description
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

#ifndef tnlGridOld3D_IMPL_H_
#define tnlGridOld3D_IMPL_H_

#ifdef HAVE_CUDA
/*template< int Dimensions, typename Real, typename Index >
__global__ void setConstantDirichletBC( const Index xSize,
                                        const Index ySize,
                                        const Real bc,
                                        Real* u );

template< int Dimensions, typename Real, typename Index >
__global__ void setDirichletBC( const Index xSize,
                                const Index ySize,
                                const Real* bc,
                                Real* u );

template< int Dimensions, typename Real, typename Index >
__global__ void setConstantNeumannBC( const Index xSize,
                                      const Index ySize,
                                      const Real hx,
                                      const Real hy,
                                      const Real bc,
                                      Real* u );

template< int Dimensions, typename Real, typename Index >
__global__ void setNeumannBC( const Index xSize,
                              const Index ySize,
                              const Real hx,
                              const Real hy,
                              const Real* bc,
                              Real* u );
*/
#endif

template< typename Real, typename Device, typename Index >
tnlGridOld< 3,Real, Device, Index > :: tnlGridOld()
{
}

template< typename Real, typename Device, typename Index >
tnlGridOld< 3,Real, Device, Index > :: tnlGridOld( const tnlString& name )
{
   this -> setName( name );
}

template< typename Real, typename Device, typename Index >
tnlGridOld< 3,Real, Device, Index > :: tnlGridOld( const tnlString& name,
                                             const tnlGridOld< 3,Real, tnlHost, Index >& grid )
{
   this -> setName( name );
   this -> setLike( grid );
}

template< typename Real, typename Device, typename Index >
tnlGridOld< 3,Real, Device, Index > :: tnlGridOld( const tnlString& name,
                                              const tnlGridOld< 3,Real, tnlCuda, Index >& grid )
{
   this -> setName( name );
   this -> setLike( grid );
}

template< typename Real, typename Device, typename Index >
const tnlStaticVector< 3,Index >& tnlGridOld< 3,Real, Device, Index > :: getDimensions() const
{
   return tnlMultiVector< 3,Real, Device, Index > :: getDimensions();
}

template< typename Real, typename Device, typename Index >
bool tnlGridOld< 3,Real, Device, Index > :: setDimensions( const tnlStaticVector< 3,Index >& dimensions )
{
   if( ! tnlMultiVector< 3,Real, Device, Index > :: setDimensions( dimensions ) )
      return false;
   spaceSteps[ 0 ] = ( domainUpperCorner[ 0 ] - domainLowerCorner[ 0 ] ) / ( Real ) ( this -> getDimensions()[ 0 ] - 1 );
   spaceSteps[ 1 ] = ( domainUpperCorner[ 1 ] - domainLowerCorner[ 1 ] ) / ( Real ) ( this -> getDimensions()[ 1 ] - 1 );
   spaceSteps[ 2 ] = ( domainUpperCorner[ 2 ] - domainLowerCorner[ 2 ] ) / ( Real ) ( this -> getDimensions()[ 2 ] - 1 );
   return true;
}

template< typename Real, typename Device, typename Index >
bool tnlGridOld< 3,Real, Device, Index > :: setDomain( const tnlStaticVector< 3,Real >& origin,
                                                     const tnlStaticVector< 3,Real >& proportions )
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
   spaceSteps[ 1 ] = ( domainUpperCorner[ 1 ] - domainLowerCorner[ 1 ] ) / ( Real ) ( this -> getDimensions()[ 1 ] - 1 );
   spaceSteps[ 2 ] = ( domainUpperCorner[ 2 ] - domainLowerCorner[ 2 ] ) / ( Real ) ( this -> getDimensions()[ 2 ] - 1 );
   return true;
}

template< typename Real, typename Device, typename Index >
   template< typename Grid >
bool tnlGridOld< 3,Real, Device, Index > :: setLike( const Grid& v )
{
   return tnlMultiVector< 3,Real, Device, Index > :: setLike( v ) &&
          this -> setDomain( v. getDomainLowerCorner(), v. getDomainUpperCorner() );
}

template< typename Real, typename Device, typename Index >
const tnlStaticVector< 3,Real >& tnlGridOld< 3,Real, Device, Index > :: getDomainLowerCorner() const
{
   return this -> domainLowerCorner;
}

template< typename Real, typename Device, typename Index >
const tnlStaticVector< 3,Real >& tnlGridOld< 3,Real, Device, Index > :: getDomainUpperCorner() const
{
   return this -> domainUpperCorner;
}

template< typename Real, typename Device, typename Index >
const tnlStaticVector< 3,Real >& tnlGridOld< 3,Real, Device, Index > :: getSpaceSteps() const
{
   return spaceSteps;
}

template< typename Real, typename Device, typename Index >
tnlString tnlGridOld< 3,Real, Device, Index > :: getType() const
{
   return tnlString( "tnlGridOld< ") +
          tnlString( "3" ) +
          tnlString( ", " ) +
          tnlString( ::getType< Real >() ) +
          tnlString( ", " ) +
          Device :: getDeviceType() +
          tnlString( ", " ) +
          tnlString( ::getType< Index >() ) +
          tnlString( " >" );
}

template< typename Real, typename Device, typename Index >
bool tnlGridOld< 3,Real, Device, Index > :: operator == ( const tnlGridOld< 3,Real, Device, Index >& grid ) const
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
   return tnlMultiVector< 3,Real, Device, Index > :: operator == ( grid );
}

template< typename Real, typename Device, typename Index >
bool tnlGridOld< 3,Real, Device, Index > :: operator != ( const tnlGridOld< 3,Real, Device, Index >& grid ) const
{
   return ! ( (* this ) == grid );
}

template< typename Real, typename Device, typename Index >
  template< typename Real2, typename Device2, typename Index2 >
tnlGridOld< 3,Real, Device, Index >&
tnlGridOld< 3,Real, Device, Index >
:: operator = ( const tnlGridOld< 3,Real2, Device2, Index2 >& grid )
{
   tnlAssert( this -> getDimensions() == grid. getDimensions(),
              cerr << "You are attempting to assign two arrays with different dimensions." << endl
                   << "First array name is " << this -> getName()
                   << " dimensions are ( " << this -> getDimensions() << " )" << endl
                   << "Second array is " << grid. getName()
                   << " dimensions are ( " << grid. getDimensions() << " )" << endl; );
   tnlVector< Real, Device, Index > :: operator = ( grid );
   return ( *this );
}

template< typename Real, typename Device, typename Index >
Real tnlGridOld< 3,Real, Device, Index > :: getValue( const tnlStaticVector< 3,Real >& point ) const
{
   tnlAssert( 0, cerr << "Interpolation is not implemented yet.");
}

template< typename Real, typename Device, typename Index >
Real tnlGridOld< 3,Real, Device, Index > :: getValue( const Real& x,
                                                             const Real& y,
                                                             const Real& z ) const
{
   return this -> getValue( tnlStaticVector< 3, Real >( x, y, z ) );
}


template< typename Real, typename Device, typename Index >
Real tnlGridOld< 3,Real, Device, Index > :: Partial_x_f( const Index i1,
                                                      const Index i2,
                                                      const Index i3 ) const
{
   tnlAssert( i1 >= 0 && i2 >= 0 && i3 >= 0 &&
              i1 < this -> getDimensions()[ tnlX ] - 1 &&
              i2 <= this -> getDimensions()[ tnlY ] - 1 &&
              i3 <= this -> getDimensions()[ tnlZ ] - 1,
              cerr << " i1 = " << i1 << " and it should be in [ 0, " <<
                      ( this -> getDimensions()[ tnlX ] - 1  ) << " ) " << endl
                   << " i2 = " << i2 << " and it should be in [ 0, " <<
                      ( this -> getDimensions()[ tnlY ] - 1  ) << " ] " << endl
                   << " i3 = " << i3 << " and it should be in [ 0, " <<
                      ( this -> getDimensions()[ tnlZ ] - 1  ) << " ] " << endl; );

   const Real& Hx = spaceSteps[ 0 ];
   tnlAssert( Hx > 0, cerr << "Hx = " << Hx << endl; );
   return ( this -> getElement( i1 + 1, i2, i3 ) -
            this ->  getElement( i1, i2, i3 ) ) / Hx;
};

template< typename Real, typename Device, typename Index >
Real tnlGridOld< 3,Real, Device, Index > :: Partial_x_b( const Index i1,
                                                      const Index i2,
                                                      const Index i3 ) const
{
   tnlAssert( i1 > 0 && i2 >= 0 && i3 >= 0 &&
              i1 <= this -> getDimensions()[ tnlX ] - 1 &&
              i2 <= this -> getDimensions()[ tnlY ] - 1 &&
              i3 <= this -> getDimensions()[ tnlZ ] - 1,
              cerr << " i1 = " << i1 << " and it should be in ( 0, " <<
                      ( this -> getDimensions()[ tnlX ] - 1  ) << " ] " << endl
                   << " i2 = " << i2 << " and it should be in [ 0, " <<
                      ( this -> getDimensions()[ tnlY ] - 1  ) << " ] " << endl
                   << " i3 = " << i3 << " and it should be in [ 0, " <<
                      ( this -> getDimensions()[ tnlZ ] - 1  ) << " ] " << endl; );

   const Real& Hx = spaceSteps[ 0 ];
   tnlAssert( Hx > 0, cerr << "Hx = " << Hx << endl; );
   return ( this -> getElement( i1, i2, i3 ) -
            this ->  getElement( i1 - 1, i2, i3 ) ) / Hx;
};

template< typename Real, typename Device, typename Index >
Real tnlGridOld< 3,Real, Device, Index > :: Partial_x( const Index i1,
                                                    const Index i2,
                                                    const Index i3 ) const
{
   tnlAssert( i1 > 0 && i2 >= 0 && i3 >= 0 &&
              i1 < this -> getDimensions()[ tnlX ] - 1 &&
              i2 <= this -> getDimensions()[ tnlY ] - 1 &&
              i3 <= this -> getDimensions()[ tnlZ ] - 1,
              cerr << " i1 = " << i1 << " and it should be in ( 0, " <<
                      ( this -> getDimensions()[ tnlX ] - 1  ) << " ) " << endl
                   << " i2 = " << i2 << " and it should be in [ 0, " <<
                      ( this -> getDimensions()[ tnlY ] - 1  ) << " ] " << endl
                   << " i3 = " << i3 << " and it should be in [ 0, " <<
                      ( this -> getDimensions()[ tnlZ ] - 1  ) << " ] " << endl; );

   const Real& Hx = spaceSteps[ 0 ];
   tnlAssert( Hx > 0, cerr << "Hx = " << Hx << endl; );
   return ( this -> getElement( i1 + 1, i2, i3 ) -
            this -> getElement( i1 - 1, i2, i3 ) ) / ( 2.0 * Hx );
};

template< typename Real, typename Device, typename Index >
Real tnlGridOld< 3,Real, Device, Index > :: Partial_xx( const Index i1,
                                                     const Index i2,
                                                     const Index i3 ) const
{
   tnlAssert( i1 > 0 && i2 >= 0 && i3 >= 0 &&
              i1 < this -> getDimensions()[ tnlX ] - 1 &&
              i2 <= this -> getDimensions()[ tnlY ] - 1 &&
              i3 <= this -> getDimensions()[ tnlZ ] - 1,
              cerr << " i1 = " << i1 << " and it should be in ( 0, " <<
                      ( this -> getDimensions()[ tnlX ] - 1  ) << " ) " << endl
                   << " i2 = " << i2 << " and it should be in [ 0, " <<
                      ( this -> getDimensions()[ tnlY ] - 1  ) << " ] " << endl
                   << " i3 = " << i3 << " and it should be in [ 0, " <<
                      ( this -> getDimensions()[ tnlZ ] - 1  ) << " ] " << endl; );

   const Real& Hx = spaceSteps[ 0 ];
   tnlAssert( Hx > 0, cerr << "Hx = " << Hx << endl; );
   return ( this -> getElement( i1 + 1, i2, i3 ) -
            2.0 * this -> getElement( i1, i2, i3 ) +
            this -> getElement( i1 - 1, i2, i3 ) ) / ( Hx * Hx );
};

template< typename Real, typename Device, typename Index >
Real tnlGridOld< 3,Real, Device, Index > :: Partial_y_f( const Index i1,
                                                      const Index i2,
                                                      const Index i3 ) const
{
   tnlAssert( i1 >= 0 && i2 >= 0 && i3 >= 0 &&
              i1 <= this -> getDimensions()[ tnlX ] - 1 &&
              i2 < this -> getDimensions()[ tnlY ] - 1 &&
              i3 <= this -> getDimensions()[ tnlZ ] - 1,
              cerr << " i1 = " << i1 << " and it should be in [ 0, " <<
                      ( this -> getDimensions()[ tnlX ] - 1  ) << " ] " << endl
                   << " i2 = " << i2 << " and it should be in [ 0, " <<
                      ( this -> getDimensions()[ tnlY ] - 1  ) << " ) " << endl
                   << " i3 = " << i3 << " and it should be in [ 0, " <<
                      ( this -> getDimensions()[ tnlZ ] - 1  ) << " ] " << endl; );

   const Real& Hy = spaceSteps[ 1 ];
   tnlAssert( Hy > 0, cerr << "Hy = " << Hy << endl; );
   return ( this -> getElement( i1, i2 + 1, i3 ) -
            this -> getElement( i1, i2, i3 ) ) / Hy;
};

template< typename Real, typename Device, typename Index >
Real tnlGridOld< 3,Real, Device, Index > :: Partial_y_b( const Index i1,
                                                      const Index i2,
                                                      const Index i3 ) const
{
   tnlAssert( i1 >= 0 && i2 > 0 && i3 >= 0 &&
              i1 <= this -> getDimensions()[ tnlX ] - 1 &&
              i2 <= this -> getDimensions()[ tnlY ] - 1 &&
              i3 <= this -> getDimensions()[ tnlZ ] - 1,
              cerr << " i1 = " << i1 << " and it should be in [ 0, " <<
                      ( this -> getDimensions()[ tnlX ] - 1  ) << " ] " << endl
                   << " i2 = " << i2 << " and it should be in ( 0, " <<
                      ( this -> getDimensions()[ tnlY ] - 1  ) << " ] " << endl
                   << " i3 = " << i3 << " and it should be in [ 0, " <<
                      ( this -> getDimensions()[ tnlZ ] - 1  ) << " ] " << endl; );

   const Real& Hy = spaceSteps[ 1 ];
   tnlAssert( Hy > 0, cerr << "Hy = " << Hy << endl; );
   return ( this -> getElement( i1, i2, i3 ) -
            this -> getElement( i1, i2 - 1, i3 ) ) / Hy;
};

template< typename Real, typename Device, typename Index >
Real tnlGridOld< 3,Real, Device, Index > :: Partial_y( const Index i1,
                                                    const Index i2,
                                                    const Index i3 ) const
{
   tnlAssert( i1 >= 0 && i2 > 0 && i3 >= 0 &&
              i1 <= this -> getDimensions()[ tnlX ] - 1 &&
              i2 < this -> getDimensions()[ tnlY ] - 1 &&
              i3 <= this -> getDimensions()[ tnlZ ] - 1,
              cerr << " i1 = " << i1 << " and it should be in [ 0, " <<
                      ( this -> getDimensions()[ tnlX ] - 1  ) << " ] " << endl
                   << " i2 = " << i2 << " and it should be in ( 0, " <<
                      ( this -> getDimensions()[ tnlY ] - 1  ) << " ) " << endl
                   << " i3 = " << i3 << " and it should be in [ 0, " <<
                      ( this -> getDimensions()[ tnlZ ] - 1  ) << " ] " << endl; );

   const Real& Hy = spaceSteps[ 1 ];
   tnlAssert( Hy > 0, cerr << "Hy = " << Hy << endl; );
   return ( this -> getElement( i1, i2 + 1, i3 ) -
            this -> getElement( i1, i2 - 1, i3 ) ) / ( 2.0 * Hy );
};

template< typename Real, typename Device, typename Index >
Real tnlGridOld< 3,Real, Device, Index > :: Partial_yy( const Index i1,
                                                     const Index i2,
                                                     const Index i3 ) const
{
   tnlAssert( i1 >= 0 && i2 > 0 && i3 >= 0 &&
              i1 <= this -> getDimensions()[ tnlX ] - 1 &&
              i2 < this -> getDimensions()[ tnlY ] - 1 &&
              i3 <= this -> getDimensions()[ tnlZ ] - 1,
              cerr << " i1 = " << i1 << " and it should be in [ 0, " <<
                      ( this -> getDimensions()[ tnlX ] - 1  ) << " ] " << endl
                   << " i2 = " << i2 << " and it should be in ( 0, " <<
                      ( this -> getDimensions()[ tnlY ] - 1  ) << " ) " << endl
                   << " i3 = " << i3 << " and it should be in [ 0, " <<
                      ( this -> getDimensions()[ tnlZ ] - 1  ) << " ] " << endl; );

   const Real& Hy = spaceSteps[ 1 ];
   tnlAssert( Hy > 0, cerr << "Hy = " << Hy << endl; );
   return ( this -> getElement( i1, i2 + 1, i3 ) -
            2.0 * this -> getElement( i1, i2, i3 ) +
            this -> getElement( i1, i2 - 1, i3 ) ) / ( Hy * Hy );
};

template< typename Real, typename Device, typename Index >
Real tnlGridOld< 3,Real, Device, Index > :: Partial_z_f( const Index i1,
                                                      const Index i2,
                                                      const Index i3 ) const
{
   tnlAssert( i1 >= 0 && i2 >= 0 && i3 >= 0 &&
              i1 <= this -> getDimensions()[ tnlX ] - 1 &&
              i2 <= this -> getDimensions()[ tnlY ] - 1 &&
              i3 < this -> getDimensions()[ tnlZ ] - 1,
              cerr << " i1 = " << i1 << " and it should be in [ 0, " <<
                      ( this -> getDimensions()[ tnlX ] - 1  ) << " ] " << endl
                   << " i2 = " << i2 << " and it should be in [ 0, " <<
                      ( this -> getDimensions()[ tnlY ] - 1  ) << " ] " << endl
                   << " i3 = " << i3 << " and it should be in [ 0, " <<
                      ( this -> getDimensions()[ tnlZ ] - 1  ) << " ) " << endl; );

   const Real& Hz = spaceSteps[ 2 ];
   tnlAssert( Hz > 0, cerr << "Hz = " << Hz << endl; );
   return ( this -> getElement( i1, i2, i3 + 1 ) -
            this -> getElement( i1, i2, i3 ) ) / Hz;
};

template< typename Real, typename Device, typename Index >
Real tnlGridOld< 3,Real, Device, Index > :: Partial_z_b( const Index i1,
                                                      const Index i2,
                                                      const Index i3 ) const
{
   tnlAssert( i1 >= 0 && i2 >= 0 && i3 > 0 &&
              i1 <= this -> getDimensions()[ tnlX ] - 1 &&
              i2 <= this -> getDimensions()[ tnlY ] - 1 &&
              i3 <= this -> getDimensions()[ tnlZ ] - 1,
              cerr << " i1 = " << i1 << " and it should be in [ 0, " <<
                      ( this -> getDimensions()[ tnlX ] - 1  ) << " ] " << endl
                   << " i2 = " << i2 << " and it should be in [ 0, " <<
                      ( this -> getDimensions()[ tnlY ] - 1  ) << " ] " << endl
                   << " i3 = " << i3 << " and it should be in ( 0, " <<
                      ( this -> getDimensions()[ tnlZ ] - 1  ) << " ] " << endl; );

   const Real& Hz = spaceSteps[ 2 ];
   tnlAssert( Hz > 0, cerr << "Hz = " << Hz << endl; );
   return ( this -> getElement( i1, i2, i3 ) -
            this -> getElement( i1, i2, i3 - 1 ) ) / Hz;
};

template< typename Real, typename Device, typename Index >
Real tnlGridOld< 3,Real, Device, Index > :: Partial_z( const Index i1,
                                                    const Index i2,
                                                    const Index i3 ) const
{
   tnlAssert( i1 >= 0 && i2 >= 0 && i3 > 0 &&
              i1 <= this -> getDimensions()[ tnlX ] - 1 &&
              i2 <= this -> getDimensions()[ tnlY ] - 1 &&
              i3 < this -> getDimensions()[ tnlZ ] - 1,
              cerr << " i1 = " << i1 << " and it should be in [ 0, " <<
                      ( this -> getDimensions()[ tnlX ] - 1  ) << " ] " << endl
                   << " i2 = " << i2 << " and it should be in [ 0, " <<
                      ( this -> getDimensions()[ tnlY ] - 1  ) << " ] " << endl
                   << " i3 = " << i3 << " and it should be in ( 0, " <<
                      ( this -> getDimensions()[ tnlZ ] - 1  ) << " ) " << endl; );

   const Real& Hz = spaceSteps[ 2 ];
   tnlAssert( Hz > 0, cerr << "Hz = " << Hz << endl; );
   return ( this -> getElement( i1, i2, i3 + 1 ) -
            this -> getElement( i1, i2, i3 - 1 ) ) / ( 2.0 * Hz );
};

template< typename Real, typename Device, typename Index >
Real tnlGridOld< 3,Real, Device, Index > :: Partial_zz( const Index i1,
                                                     const Index i2,
                                                     const Index i3 ) const
{
   tnlAssert( i1 >= 0 && i2 >= 0 && i3 > 0 &&
              i1 <= this -> getDimensions()[ tnlX ] - 1 &&
              i2 <= this -> getDimensions()[ tnlY ] - 1 &&
              i3 < this -> getDimensions()[ tnlZ ] - 1,
              cerr << " i1 = " << i1 << " and it should be in [ 0, " <<
                      ( this -> getDimensions()[ tnlX ] - 1  ) << " ] " << endl
                   << " i2 = " << i2 << " and it should be in [ 0, " <<
                      ( this -> getDimensions()[ tnlY ] - 1  ) << " ] " << endl
                   << " i3 = " << i3 << " and it should be in ( 0, " <<
                      ( this -> getDimensions()[ tnlZ ] - 1  ) << " ) " << endl; );

   const Real& Hz = spaceSteps[ 2 ];
   tnlAssert( Hz > 0, cerr << "Hz = " << Hz << endl; );
   return ( this -> getElement( i1, i2, i3 + 1 ) -
            2.0 * this -> getElement( i1, i2, i3 ) +
            this -> getElement( i1, i2, i3 - 1 ) ) / ( Hz * Hz );
};

template< typename Real, typename Device, typename Index >
void tnlGridOld< 3,Real, Device, Index > :: setDirichletBC( const tnlGridOld< 3,Real, Device, Index >&bc,
                                                         const tnlStaticVector< 3,bool >& lowerBC,
                                                         const tnlStaticVector< 3,bool >& upperBC )
{
   if( Device :: getDevice() == tnlHostDevice )
   {
      const Index xSize = this -> getDimensions(). x();
      const Index ySize = this -> getDimensions(). y();
      const Index zSize = this -> getDimensions(). z();
      Index i, j, k;

      if( lowerBC. x() )
         for( i = 0; i < xSize; i ++ )
            for( k = 0; k < zSize; k ++ )
               ( *this )( i, 0, k ) = bc( i, 0, k );

      if( upperBC. x() )
         for( i = 0; i < xSize; i ++ )
            for( k = 0; k < zSize; k ++ )
               ( *this )( i, ySize - 1, k ) = ( i, ySize - 1, k );

      if( lowerBC. y() )
         for( j = 0; j < ySize; j ++ )
            for( k = 0; k < zSize; k ++ )
               ( *this )( 0, j, k ) = bc( 0, j, k );

      if( upperBC. y() )
         for( j = 0; j < ySize; j ++ )
            for( k = 0; k < zSize; k ++ )
               ( *this )( xSize - 1, j, k ) = bc( xSize - 1, j, k );

      if( lowerBC. z() )
         for( i = 0; i < xSize; i ++ )
            for( j = 0; j < ySize; j ++ )
               ( *this )( i, j, 0 ) = bc( i, j, 0 );

      if( upperBC. z() )
         for( i = 0; i < xSize; i ++ )
            for( j = 0; j < ySize; j ++ )
               ( *this )( i, j, zSize - 1 ) = bc( i, j, zSize - 1 );
   }
   if( Device :: getDevice() == tnlCudaDevice )
   {
#ifdef HAVE_CUDA
      tnlAssert( false, );
      /*const Index size = this -> getSize();
      const Index desBlockSize = 64;
      const Index gridSize = size / desBlockSize + ( size % desBlockSize != 0 );
      dim3 gridDim( 0 ), blockDim( 0 );
      gridDim. x = gridSize;
      blockDim. x = desBlockSize;
      :: setDirichletBC< 3,Real, Index ><<< gridDim, blockDim >>>( this -> getDimensions(). x(),
                                                                    this -> getDimensions(). y(),
                                                                    bc. getData(),
                                                                    this -> getData() );*/

#endif
   }
}

template< typename Real, typename Device, typename Index >
void tnlGridOld< 3,Real, Device, Index > :: setDirichletBC( const Real& bcValue,
                                                         const tnlStaticVector< 3,bool >& lowerBC,
                                                         const tnlStaticVector< 3,bool >& upperBC )
{
   if( Device :: getDevice() == tnlHostDevice )
   {
      const Index xSize = this -> getDimensions(). x();
      const Index ySize = this -> getDimensions(). y();
      const Index zSize = this -> getDimensions(). z();
      Index i, j, k;

      if( lowerBC. x() )
         for( i = 0; i < xSize; i ++ )
            for( k = 0; k < zSize; k ++ )
               ( *this )( i, 0, k ) = bcValue;

      if( upperBC. x() )
         for( i = 0; i < xSize; i ++ )
            for( k = 0; k < zSize; k ++ )
               ( *this )( i, ySize - 1, k ) = bcValue;

      if( lowerBC. y() )
         for( j = 0; j < ySize; j ++ )
            for( k = 0; k < zSize; k ++ )
               ( *this )( 0, j, k ) = bcValue;

      if( upperBC. y() )
         for( j = 0; j < ySize; j ++ )
            for( k = 0; k < zSize; k ++ )
               ( *this )( xSize - 1, j, k ) = bcValue;

      if( lowerBC. z() )
         for( i = 0; i < xSize; i ++ )
            for( j = 0; j < ySize; j ++ )
               ( *this )( i, j, 0 ) = bcValue;

      if( upperBC. z() )
         for( i = 0; i < xSize; i ++ )
            for( j = 0; j < ySize; j ++ )
               ( *this )( i, j, zSize - 1 ) =  bcValue;
   }
   if( Device :: getDevice() == tnlCudaDevice )
   {
#ifdef HAVE_CUDA
      const Index size = this -> getSize();
      const Index desBlockSize = 64;
      const Index gridSize = size / desBlockSize + ( size % desBlockSize != 0 );
      dim3 gridDim( 0 ), blockDim( 0 );
      gridDim. x = gridSize;
      blockDim. x = desBlockSize;
      :: setConstantDirichletBC< 3,Real, Index ><<< gridDim, blockDim >>>( this -> getDimensions(). x(),
                                                                            this -> getDimensions(). y(),
                                                                            bcValue,
                                                                            this -> getData() );
#endif
   }
}

template< typename Real, typename Device, typename Index >
void tnlGridOld< 3,Real, Device, Index > :: setNeumannBC( const tnlGridOld< 3,Real, Device, Index >&bc,
                                                       const tnlStaticVector< 3,bool >& lowerBC,
                                                       const tnlStaticVector< 3,bool >& upperBC )
{
   if( Device :: getDevice() == tnlHostDevice )
   {
      const Real& hx = this ->  getSpaceSteps(). x();
      const Real& hy = this ->  getSpaceSteps(). y();
      const Real& hz = this ->  getSpaceSteps(). z();

      const Index xSize = this -> getDimensions(). x();
      const Index ySize = this -> getDimensions(). y();
      const Index zSize = this -> getDimensions(). z();
      Index i, j, k;

      if( lowerBC. x() )
         for( i = 0; i < xSize; i ++ )
            for( k = 0; k < zSize; k ++ )
               ( *this )( i, 0, k ) = ( *this )( i, 1, k ) + hy * bc. getElement( i, 0, k );
      if( upperBC. x() )
         for( i = 0; i < xSize; i ++ )
            for( k = 0; k < zSize; k ++ )
               ( *this )( i, ySize - 1, k ) = ( *this )( i, ySize - 2, k ) + hy * bc( i, ySize - 1, k );

      if( lowerBC. y() )
         for( j = 0; j < ySize; j ++ )
            for( k = 0; k < zSize; k ++ )
               ( *this )( 0, j, k ) =  ( *this )( 1, j, k ) + hx * bc( 0, j, k );

      if( upperBC. y() )
         for( j = 0; j < ySize; j ++ )
            for( k = 0; k < zSize; k ++ )
               ( *this )( xSize - 1, j, k ) = ( *this )( xSize - 2, j, k ) + hx * bc( xSize - 1, j, k );

      if( lowerBC. z() )
         for( i = 0; i < xSize; i ++ )
            for( j = 0; j < ySize; j ++ )
               ( *this )( i, j, 0 ) = ( *this )( i, j, 1 ) + hz * bc( i, j, 0 );

      if( upperBC. z() )
         for( i = 0; i < xSize; i ++ )
            for( j = 0; j < ySize; j ++ )
               ( *this )( i, j, zSize - 1 ) = ( *this )( i, j, zSize - 2 ) + hz * bc( i, j, zSize - 1 );
   }
   if( Device :: getDevice() == tnlCudaDevice )
   {
#ifdef HAVE_CUDA
      /*
      const Index size = this -> getSize();
      const Index desBlockSize = 64;
      const Index gridSize = size / desBlockSize + ( size % desBlockSize != 0 );
      dim3 gridDim( 0 ), blockDim( 0 );
      gridDim. x = gridSize;
      blockDim. x = desBlockSize;
      ::setNeumannBC< 3,Real, Index ><<< gridDim, blockDim >>>( this -> getDimensions(). x(),
                                                                 this -> getDimensions(). y(),
                                                                 this -> getSpaceSteps(). x(),
                                                                 this -> getSpaceSteps(). y(),
                                                                 bc. getData(),
                                                                 this -> getData() );
                                                                 */
#endif
   }
}

template< typename Real, typename Device, typename Index >
void tnlGridOld< 3,Real, Device, Index > :: setNeumannBC( const Real& bcValue,
                                                       const tnlStaticVector< 3,bool >& lowerBC,
                                                       const tnlStaticVector< 3,bool >& upperBC )
{
   if( Device :: getDevice() == tnlHostDevice )
   {
      const Real& hx = this -> getSpaceSteps(). x();
      const Real& hy = this -> getSpaceSteps(). y();
      const Real& hz = this -> getSpaceSteps(). z();

      const Index xSize = this -> getDimensions(). x();
      const Index ySize = this -> getDimensions(). y();
      const Index zSize = this -> getDimensions(). z();
      Index i, j, k;

      if( lowerBC. x() )
         for( i = 0; i < xSize; i ++ )
            for( k = 0; k < zSize; k ++ )
               ( *this )( i, 0, k ) = ( *this )( i, 1, k ) + hy * bcValue;
      if( upperBC. x() )
         for( i = 0; i < xSize; i ++ )
            for( k = 0; k < zSize; k ++ )
               ( *this )( i, ySize - 1, k ) = ( *this )( i, ySize - 2, k ) + hy * bcValue;

      if( lowerBC. y() )
         for( j = 0; j < ySize; j ++ )
            for( k = 0; k < zSize; k ++ )
               ( *this )( 0, j, k ) = ( *this )( 1, j, k ) + hx * bcValue;

      if( upperBC. y() )
         for( j = 0; j < ySize; j ++ )
            for( k = 0; k < zSize; k ++ )
               ( *this )( xSize - 1, j, k ) = ( *this )( xSize - 2, j, k ) + hx * bcValue;

      if( lowerBC. z() )
         for( i = 0; i < xSize; i ++ )
            for( j = 0; j < ySize; j ++ )
               ( *this )( i, j, 0 ) = ( *this )( i, j, 1 ) + hz * bcValue;

      if( upperBC. z() )
         for( i = 0; i < xSize; i ++ )
            for( j = 0; j < ySize; j ++ )
               ( *this )( i, j, zSize - 1 ) = ( *this )( i, j, zSize - 2 ) + hz * bcValue;
   }
   if( Device :: getDevice() == tnlCudaDevice )
   {
#ifdef HAVE_CUDA
      const Index size = this -> getSize();
      const Index desBlockSize = 64;
      const Index gridSize = size / desBlockSize + ( size % desBlockSize != 0 );
      dim3 gridDim( 0 ), blockDim( 0 );
      gridDim. x = gridSize;
      blockDim. x = desBlockSize;
      :: setConstantNeumannBC< 3,Real, Index ><<< gridDim, blockDim >>>( this -> getDimensions(). x(),
                                                                          this -> getDimensions(). y(),
                                                                          this -> getSpaceSteps(). x(),
                                                                          this -> getSpaceSteps(). y(),
                                                                          bcValue,
                                                                          this -> getData() );
#endif
   }
}

/*template< typename Real, typename Device, typename Index >
Real tnlGridOld< 3,Real, Device, Index > :: getMax() const
{
   return tnlMax( * this );
}

template< typename Real, typename Device, typename Index >
Real tnlGridOld< 3,Real, Device, Index > :: getMin() const
{
   return tnlMin( * this );
}

template< typename Real, typename Device, typename Index >
Real tnlGridOld< 3,Real, Device, Index > :: getAbsMax() const
{
   return tnlAbsMax( * this );
}

template< typename Real, typename Device, typename Index >
Real tnlGridOld< 3,Real, Device, Index > :: getAbsMin() const
{
   return tnlAbsMin( * this );
}*/

template< typename Real, typename Device, typename Index >
Real tnlGridOld< 3,Real, Device, Index > :: getLpNorm( const Real& p ) const
{
   Real result = this -> lpNorm( p );
   return result * getSpaceSteps(). x()
                 * getSpaceSteps(). y()
                 * getSpaceSteps(). z();
}

/*
template< typename Real, typename Device, typename Index >
Real tnlGridOld< 3,Real, Device, Index > :: getSum() const
{
   return tnlMultiVector< 3,Real, Device, Index > :: sum( * this );
}

template< typename Real, typename Device, typename Index >
Real tnlGridOld< 3,Real, Device, Index > :: getDifferenceMax( const tnlVector< Real, Device, Index >& v ) const
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
Real tnlGridOld< 3,Real, Device, Index > :: getDifferenceMin( const tnlVector< Real, Device, Index >& v ) const
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
Real tnlGridOld< 3,Real, Device, Index > :: getDifferenceAbsMax( const tnlVector< Real, Device, Index >& v ) const
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
Real tnlGridOld< 3,Real, Device, Index > :: getDifferenceAbsMin( const tnlVector< Real, Device, Index >& v ) const
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
}*/

template< typename Real, typename Device, typename Index >
Real tnlGridOld< 3,Real, Device, Index > :: getDifferenceLpNorm( const tnlVector< Real, Device, Index >& v, const Real& p ) const
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
   return result * getSpaceSteps(). x()
                 * getSpaceSteps(). y()
                 * getSpaceSteps(). z();
}

/*template< typename Real, typename Device, typename Index >
Real tnlGridOld< 3,Real, Device, Index > :: getDifferenceSum( const tnlVector< Real, Device, Index >& v ) const
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
void tnlGridOld< 3,Real, Device, Index > :: scalarMultiplication( const Real& alpha )
{
   return this -> scalarMultiplication( alpha );
}

template< typename Real, typename Device, typename Index >
Real tnlGridOld< 3,Real, Device, Index > :: scalarProduct( const tnlVector< Real, Device, Index >& v ) const
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
void tnlGridOld< 3,Real, Device, Index > :: alphaXPlusY( const Real& alpha,
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
   this -> alphaXPlusY( alpha, x );
};

template< typename Real, typename Device, typename Index >
void tnlGridOld< 3,Real, Device, Index > :: saxmy( const Real& alpha,
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
};*/


template< typename Real, typename Device, typename Index >
bool tnlGridOld< 3,Real, Device, Index > :: save( tnlFile& file ) const
{
   if( ! tnlMultiVector< 3,Real, Device, Index > :: save( file ) )
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
bool tnlGridOld< 3,Real, Device, Index > :: load( tnlFile& file )
{
   if( ! tnlMultiVector< 3,Real, Device, Index > :: load( file ) )
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
bool tnlGridOld< 3,Real, Device, Index > :: save( const tnlString& fileName ) const
{
   return tnlObject :: save( fileName );
}

template< typename Real, typename Device, typename Index >
bool tnlGridOld< 3,Real, Device, Index > :: load( const tnlString& fileName )
{
   return tnlObject :: load( fileName );
}

template< typename Real, typename Device, typename Index >
bool tnlGridOld< 3,Real, Device, Index > :: draw( const tnlString& fileName,
                                                         const tnlString& format,
                                                         const tnlStaticVector< 3,Index > steps ) const
{
   tnlAssert( steps > ( tnlStaticVector< 3,Index >( 0 ) ),
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
   const Index xSize = this -> getDimensions()[ tnlX ];
   const Index ySize = this -> getDimensions()[ tnlY ];
   const Index zSize = this -> getDimensions()[ tnlZ ];
   const Real& ax = this -> getDomainLowerCorner()[ tnlX ];
   const Real& ay = this -> getDomainLowerCorner()[ tnlY ];
   const Real& az = this -> getDomainLowerCorner()[ tnlZ ];
   const Real& hx = this -> getSpaceSteps()[ tnlX ];
   const Real& hy = this -> getSpaceSteps()[ tnlY ];
   const Real& hz = this -> getSpaceSteps()[ tnlZ ];
   if( format == "gnuplot" )
   {
      cout << "GNUPLOT is not supported for tnlGridOld3D." << endl;
      return false;
   }
   if( format == "vti" )
   {
      file << "<VTKFile type=\"ImagegetString\" version=\"0.1\" byte_order=\"LittleEndian\">" << endl;
      file << "<ImagegetString WholeExtent=\""
           << "0 " << xSize - 1
           << " 0 " << ySize - 1
           << " 0 " << zSize - 1 << "\" Origin=\"0 0 0\" Spacing=\""
           << hx * steps[ tnlX ] << " " << hy * steps[ tnlY ] << " " << hz * steps[ tnlZ ] << "\">" << endl;
      file << "<Piece Extent=\"0 "
           << xSize - 1 << " 0 "
           << ySize - 1 << " 0 "
           << zSize - 1 << "\">" << endl;
      file << "<PointgetString Scalars=\"order_parameter\">" << endl;
      file << "<getStringArray Name=\"order_parameter\" type=\"Float32\" format=\"ascii\">" << endl;
      file. flags( ios_base::scientific );
      Index iStep = steps[ tnlX ];
      Index jStep = steps[ tnlY ];
      Index kStep = steps[ tnlZ ];
      for( Index k = 0; k <= zSize - kStep; k += kStep )
         for( Index j = 0; j <= ySize - jStep; j += jStep )
            for( Index i = 0; i <= xSize - iStep; i += iStep )
            {
              file << this -> getElement( i, j, k ) << " ";
            }
      file << endl;
      file << "</getStringArray>" << endl;
      file << "</PointgetString>" << endl;
      file << "</Piece>" << endl;
      file << "</ImagegetString>" << endl;
      file << "</VTKFile>" << endl;
      return true;
   }
   if( format == "povray" )
   {
      file. put( ( char ) ( xSize >> 8 ) );
      file. put( ( char ) ( xSize & 0xff ) );
      file. put( ( char ) ( ySize >> 8 ) );
      file. put( ( char ) ( ySize & 0xff ) );
      file. put( ( char ) ( zSize >> 8 ) );
      file. put( ( char ) ( zSize & 0xff ) );
      Real min( this -> getElement( 0, 0, 0 ) ), max( this -> getElement( 0, 0, 0 ) );
      for( Index k = 0; k < zSize; k ++ )
         for( Index j = 0; j < ySize; j ++ )
            for( Index i = 0; i < xSize; i ++ )
            {
               min = Min( min, this -> getElement( i, j, k ) );
               max = Max( max, this -> getElement( i, j, k ) );
            }

      for( Index k = 0; k < zSize; k ++ )
         for( Index j = 0; j < ySize; j ++ )
            for( Index i = 0; i < xSize; i ++ )
            {
               int v = Real( 255.0 ) * ( this -> getElement( i, j, k ) - min ) / ( max - min );
               file. write( ( char* ) &v, sizeof( int ) );
            }
      return true;
   }
   cerr << endl << "I do not know a format " << format << " for tnlGridOld with 3 dimensions.";
   return false;
}

#ifdef HAVE_CUDA
/*
template< int Dimensions, typename Real, typename Index >
__global__ void  setConstantDirichletBC( const Index xSize,
                                         const Index ySize,
                                         const Real bc,
                                         Real* u )
{
   const Index ij = blockIdx. x * blockDim. x + threadIdx. x;
   const Index i = ij / ySize;
   const Index j = ij % ySize;

   if( ij < xSize * ySize &&
       ( i == 0 || j == 0 || i == xSize - 1 || j == ySize - 1 ) )
   {
      u[ ij ] = bc;
   }

}

template< int Dimensions, typename Real, typename Index >
__global__ void setDirichletBC( const Index xSize,
                                const Index ySize,
                                const Real* bc,
                                Real* u )
{
   const Index ij = blockIdx. x * blockDim. x + threadIdx. x;
   const Index i = ij / ySize;
   const Index j = ij % ySize;

   if( ij < xSize * ySize &&
       ( i == 0 || j == 0 || i == xSize - 1 || j == ySize - 1 ) )
   {
      u[ ij ] = bc[ ij ];
   }
}

template< int Dimensions, typename Real, typename Index >
__global__ void setConstantNeumannBC( const Index xSize,
                                      const Index ySize,
                                      const Real hx,
                                      const Real hy,
                                      const Real bc,
                                      Real* u )
{
   const Index ij = blockIdx. x * blockDim. x + threadIdx. x;
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
   }
}

template< int Dimensions, typename Real, typename Index >
__global__ void setNeumannBC( const Index xSize,
                              const Index ySize,
                              const Real hx,
                              const Real hy,
                              const Real* bc,
                              Real* u )
{
   const Index ij = blockIdx. x * blockDim. x + threadIdx. x;
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
   }
}
*/
#endif




#endif /* tnlGrid3D_IMPL_H_ */
