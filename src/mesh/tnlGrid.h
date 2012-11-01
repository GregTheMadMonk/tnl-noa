/***************************************************************************
                          tnlGrid.h  -  description
                             -------------------
    begin                : Dec 12, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
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

#ifndef TNLGRID_H_
#define TNLGRID_H_

#include <iomanip>
#include <fstream>
#include <core/tnlAssert.h>
#include <core/tnlMultiArray.h>
#include <core/tnlVector.h>

using namespace std;

template< int Dimensions, typename Real = double, typename Device = tnlHost, typename Index = int >
class tnlGrid : public tnlMultiArray< Dimensions, Real, Device, Index >
{
   //! We do not allow constructor without parameters.
   tnlGrid();

   //! We do not allow copy constructor without object name.
   tnlGrid( const tnlGrid< Dimensions, Real, Device, Index >& a );

   public:

   tnlGrid( const tnlString& name );

   tnlGrid( const tnlString& name,
            const tnlGrid< Dimensions, Real, tnlHost, Index >& grid );

   tnlGrid( const tnlString& name,
            const tnlGrid< Dimensions, Real, tnlCuda, Index >& grid );

   const tnlTuple< Dimensions, Index >& getDimensions() const;

   //! Sets the dimensions
   /***
    * This method also must recompute space steps. It is save to call setDimensions and
    * setDomain in any order. Both recompute the space steps.
    */
   bool setDimensions( const tnlTuple< Dimensions, Index >& dimensions );

   //! Sets the computation domain in form of "rectangle".
   /***
    * This method also must recompute space steps. It is save to call setDimensions and
    * setDomain in any order. Both recompute the space steps.
    */

   bool setDomain( const tnlTuple< Dimensions, Real >& lowerCorner,
                   const tnlTuple< Dimensions, Real >& upperCorner );

   //! Set dimensions and domain of the grid using another grid as a template
   bool setLike( const tnlGrid< Dimensions, Real, tnlHost, Index >& v );

   //! Set dimensions and domain of the grid using another grid as a template
   bool setLike( const tnlGrid< Dimensions, Real, tnlCuda, Index >& v );

   const tnlTuple< Dimensions, Real >& getDomainLowerCorner() const;

   const tnlTuple< Dimensions, Real >& getDomainUpperCorner() const;

   const tnlTuple< Dimensions, Real >& getSpaceSteps() const;

   tnlString getType() const;

   bool operator == ( const tnlGrid< Dimensions, Real, Device, Index >& array ) const;

   bool operator != ( const tnlGrid< Dimensions, Real, Device, Index >& array ) const;

   template< typename Real2, typename Device2, typename Index2 >
   tnlGrid< Dimensions, Real, Device, Index >& operator = ( const tnlGrid< Dimensions, Real2, Device2, Index2 >& array );

   //! This method interpolates value at given point.
   Real getValue( const tnlTuple< Dimensions, Real >& point ) const;

   //! Interpolation for 1D grid.
   Real getValue( const Real& x ) const;

   //! Interpolation for 2D grid.
   Real getValue( const Real& x,
                  const Real& y ) const;

   //! Interpolation for 3D grid.
   Real getValue( const Real& x,
                  const Real& y,
                  const Real& z ) const;

   //! Forward difference w.r.t x
   Real Partial_x_f( const Index i1 ) const;

   //! Forward difference w.r.t x in two dimensions
   Real Partial_x_f( const Index i1,
                     const Index i2 ) const;

   //! Forward difference w.r.t x in three dimensions
   Real Partial_x_f( const Index i1,
                     const Index i2,
                     const Index i3 ) const;

   //! Backward difference w.r.t x
   Real Partial_x_b( const Index i1 ) const;

   //! Backward difference w.r.t x in two dimensions
   Real Partial_x_b( const Index i1,
                     const Index i2 ) const;

   //! Backward difference w.r.t x in three dimensions
   Real Partial_x_b( const Index i1,
                     const Index i2,
                     const Index i3 ) const;

   //! Central difference w.r.t. x
   Real Partial_x( const Index i1 ) const;

   //! Central difference w.r.t. x in two dimensions
   Real Partial_x( const Index i1,
                   const Index i2 ) const;

   //! Central difference w.r.t. x
   Real Partial_x( const Index i1,
                   const Index i2,
                   const Index i3 ) const;

   //! Second order difference w.r.t. x
   Real Partial_xx( const Index i1 ) const;

   //! Second order difference w.r.t. x in two dimensions
   Real Partial_xx( const Index i1,
                    const Index i2 ) const;

   //! Second order difference w.r.t. x
   Real Partial_xx( const Index i1,
                    const Index i2,
                    const Index i3 ) const;

   //! Forward difference w.r.t y
   Real Partial_y_f( const Index i1,
                     const Index i2 ) const;

   //! Forward difference w.r.t y in three dimensions
   Real Partial_y_f( const Index i1,
                     const Index i2,
                     const Index i3 ) const;

   //! Backward difference w.r.t y
   Real Partial_y_b( const Index i1,
                     const Index i2 ) const;

   //! Backward difference w.r.t y in three dimensions
   Real Partial_y_b( const Index i1,
                     const Index i2,
                     const Index i3 ) const;

   //! Central difference w.r.t y
   Real Partial_y( const Index i1,
                   const Index i2 ) const;

   //! Central difference w.r.t y
   Real Partial_y( const Index i1,
                   const Index i2,
                   const Index i3 ) const;

   //! Second order difference w.r.t. y
   Real Partial_yy( const Index i1,
                    const Index i2 ) const;

   //! Second order difference w.r.t. y in three dimensions
   Real Partial_yy( const Index i1,
                    const Index i2,
                    const Index i3 ) const;

   //! Forward difference w.r.t z
   Real Partial_z_f( const Index i1,
                     const Index i2,
                     const Index i3 ) const;

   //! Backward difference w.r.t z
   Real Partial_z_b( const Index i1,
                     const Index i2,
                     const Index i3 ) const;

   //! Central difference w.r.t z
   Real Partial_z( const Index i1,
                   const Index i2,
                   const Index i3 ) const;

   //! Second order difference w.r.t. z
   Real Partial_zz( const Index i1,
                    const Index i2,
                    const Index i3 ) const;

   //! Set space dependent Dirichlet boundary conditions
   void setDirichletBC( const tnlGrid< Dimensions, Real, Device, Index >&bc,
                        const tnlTuple< Dimensions, bool >& lowerBC,
                        const tnlTuple< Dimensions, bool >& upperBC );

   //! Set constant Dirichlet boundary conditions
   void setDirichletBC( const Real& bc,
                        const tnlTuple< Dimensions, bool >& lowerBC,
                        const tnlTuple< Dimensions, bool >& upperBC );

   //! Set space dependent Neumann boundary conditions
   void setNeumannBC( const tnlGrid< Dimensions, Real, Device, Index >&bc,
                      const tnlTuple< Dimensions, bool >& lowerBC,
                      const tnlTuple< Dimensions, bool >& upperBC );

   //! Set constant Neumann boundary conditions
   void setNeumannBC( const Real& bc,
                      const tnlTuple< Dimensions, bool >& lowerBC,
                      const tnlTuple< Dimensions, bool >& upperBC );

   Real getMax() const;

   Real getMin() const;

   Real getAbsMax() const;

   Real getAbsMin() const;

   Real getLpNorm() const;

   Real getSum() const;

   Real getDifferenceMax( const tnlVector< Real, Device, Index >& v ) const;

   Real getDifferenceMin( const tnlVector< Real, Device, Index >& v ) const;

   Real getDifferenceAbsMax( const tnlVector< Real, Device, Index >& v ) const;

   Real getDifferenceAbsMin( const tnlVector< Real, Device, Index >& v ) const;

   Real getDifferenceLpNorm( const tnlVector< Real, Device, Index >& v, const Real& p ) const;

   Real getDifferenceSum( const tnlVector< Real, Device, Index >& v ) const;

   void scalarMultiplication( const Real& alpha );

   //! Compute scalar dot product
   Real sdot( const tnlVector< Real, Device, Index >& v ) const;

   //! Compute SAXPY operation (Scalar Alpha X Pus Y ).
   void saxpy( const Real& alpha,
                const tnlVector< Real, Device, Index >& x );

   //! Compute SAXMY operation (Scalar Alpha X Minus Y ).
   /*!**
    * It is not a standart BLAS function but is useful for GMRES solver.
    */
   void saxmy( const Real& alpha,
                const tnlVector< Real, Device, Index >& x );

   //! Method for saving the object to a file as a binary data
   bool save( tnlFile& file ) const;

   //! Method for restoring the object from a file
   bool load( tnlFile& file );

   bool save( const tnlString& fileName ) const;

   bool load( const tnlString& fileName );

   //! This method writes the grid in some format suitable for some other preprocessing.
   /*! Possible formats are:
    *  1) Gnuplot format (gnuplot)
    *  2) VTI format (vti)
    *  3) Povray format (povray) - only for 3D.
    */
   bool draw( const tnlString& fileName,
              const tnlString& format,
              const tnlTuple< Dimensions, Index > steps = ( tnlTuple< Dimensions, Index > ) 1 ) const;

   protected:
   tnlTuple< Dimensions, Real > domainLowerCorner, domainUpperCorner, spaceSteps;
};

#ifdef HAVE_CUDA
template< int Dimensions, typename Real, typename Index >
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
#endif

template< int Dimensions, typename Real, typename Device, typename Index >
tnlGrid< Dimensions, Real, Device, Index > :: tnlGrid( const tnlString& name )
: tnlMultiArray< Dimensions, Real, Device, Index >( name )
  {
  }

template< int Dimensions, typename Real, typename Device, typename Index >
tnlGrid< Dimensions, Real, Device, Index > :: tnlGrid( const tnlString& name,
                                                       const tnlGrid< Dimensions, Real, tnlHost, Index >& grid )
: tnlMultiArray< Dimensions, Real, Device, Index >( name, grid )
{
   this -> setDomain( grid. getDomainLowerCorner(),
                      grid. getDomainUpperCorner() );
}

template< int Dimensions, typename Real, typename Device, typename Index >
tnlGrid< Dimensions, Real, Device, Index > :: tnlGrid( const tnlString& name,
                                                       const tnlGrid< Dimensions, Real, tnlCuda, Index >& grid )
: tnlMultiArray< Dimensions, Real, Device, Index >( name, grid )
{
   this -> setDomain( grid. getDomainLowerCorner(),
                      grid. getDomainUpperCorner() );
}

template< int Dimensions, typename Real, typename Device, typename Index >
const tnlTuple< Dimensions, Index >& tnlGrid< Dimensions, Real, Device, Index > :: getDimensions() const
{
   return tnlMultiArray< Dimensions, Real, Device, Index > :: getDimensions();
}

template< int Dimensions, typename Real, typename Device, typename Index >
bool tnlGrid< Dimensions, Real, Device, Index > :: setDimensions( const tnlTuple< Dimensions, Index >& dimensions )
{
   if( ! tnlMultiArray< Dimensions, Real, Device, Index > :: setDimensions( dimensions ) )
      return false;
   for( int i = 0; i < Dimensions; i ++ )
      spaceSteps[ i ] = ( domainUpperCorner[ i ] - domainLowerCorner[ i ] ) / ( Real ) ( this -> getDimensions()[ i ] - 1 );
   return true;
}

template< int Dimensions, typename Real, typename Device, typename Index >
bool tnlGrid< Dimensions, Real, Device, Index > :: setDomain( const tnlTuple< Dimensions, Real >& lowerCorner,
                                                              const tnlTuple< Dimensions, Real >& upperCorner )
{
   if( lowerCorner >= upperCorner )
   {
      cerr << "Wrong parameters for the grid domain of " << this -> getName() << ". The low corner must by smaller than the high corner." << endl
               << "lowerCorner = " << lowerCorner << endl << "upperCorner = " << upperCorner << endl;
      return false;
   }
   domainLowerCorner = lowerCorner;
   domainUpperCorner = upperCorner;
   for( int i = 0; i < Dimensions; i ++ )
      spaceSteps[ i ] = ( domainUpperCorner[ i ] - domainLowerCorner[ i ] ) / ( Real ) ( this -> getDimensions()[ i ] - 1 );
   return true;
}

template< int Dimensions, typename Real, typename Device, typename Index >
bool tnlGrid< Dimensions, Real, Device, Index > :: setLike( const tnlGrid< Dimensions, Real, tnlHost, Index >& v )
{
   return tnlMultiArray< Dimensions, Real, Device, Index > :: setLike( v ) &&
          this -> setDomain( v. getDomainLowerCorner(), v. getDomainUpperCorner() );
}

template< int Dimensions, typename Real, typename Device, typename Index >
bool tnlGrid< Dimensions, Real, Device, Index > :: setLike( const tnlGrid< Dimensions, Real, tnlCuda, Index >& v )
{
   return tnlMultiArray< Dimensions, Real, Device, Index > :: setLike( v ) &&
          this -> setDomain( v. getDomainLowerCorner(), v. getDomainUpperCorner() );
}

template< int Dimensions, typename Real, typename Device, typename Index >
const tnlTuple< Dimensions, Real >& tnlGrid< Dimensions, Real, Device, Index > :: getDomainLowerCorner() const
{
   return this -> domainLowerCorner;
}

template< int Dimensions, typename Real, typename Device, typename Index >
const tnlTuple< Dimensions, Real >& tnlGrid< Dimensions, Real, Device, Index > :: getDomainUpperCorner() const
{
   return this -> domainUpperCorner;
}

template< int Dimensions, typename Real, typename Device, typename Index >
const tnlTuple< Dimensions, Real >& tnlGrid< Dimensions, Real, Device, Index > :: getSpaceSteps() const
{
   return spaceSteps;
}

template< int Dimensions, typename Real, typename Device, typename Index >
tnlString tnlGrid< Dimensions, Real, Device, Index > :: getType() const
{
   return tnlString( "tnlGrid< ") +
          tnlString( Dimensions ) +
          tnlString( ", " ) +
          tnlString( getParameterType< Real >() ) +
          tnlString( ", " ) +
          Device :: getDeviceType() +
          tnlString( ", " ) +
          tnlString( getParameterType< Index >() ) +
          tnlString( " >" );
}

template< int Dimensions, typename Real, typename Device, typename Index >
bool tnlGrid< Dimensions, Real, Device, Index > :: operator == ( const tnlGrid< Dimensions, Real, Device, Index >& grid ) const
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
   return tnlMultiArray< Dimensions, Real, Device, Index > :: operator == ( grid );
}

template< int Dimensions, typename Real, typename Device, typename Index >
bool tnlGrid< Dimensions, Real, Device, Index > :: operator != ( const tnlGrid< Dimensions, Real, Device, Index >& grid ) const
{
   return ! ( (* this ) == grid );
}

template< int Dimensions, typename Real, typename Device, typename Index >
  template< typename Real2, typename Device2, typename Index2 >
tnlGrid< Dimensions, Real, Device, Index >&
tnlGrid< Dimensions, Real, Device, Index >
:: operator = ( const tnlGrid< Dimensions, Real2, Device2, Index2 >& grid )
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

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: getValue( const tnlTuple< Dimensions, Real >& point ) const
{
   if( Dimensions == 1)
   {
      tnlAssert( 0, cerr << "Interpolation is not implemented yet.");
   }
   if( Dimensions == 2 )
   {
      Real x = ( point[ 0 ] - domainLowerCorner[ 0 ] ) / spaceSteps[ 0 ];
      Real y = ( point[ 1 ] - domainLowerCorner[ 1 ] ) / spaceSteps[ 1 ];
      Index ix = ( Index ) ( x );
      Index iy = ( Index ) ( y );
      Real dx = x - ( Real ) ix;
      Real dy = y - ( Real ) iy;
      if( iy >= this -> getDimensions()[ tnlY ] - 1 )
      {
         if( ix >= this -> getDimensions()[ tnlX ] - 1 )
            return  this -> getElement( this -> getDimensions()[ tnlX ] - 1,
                                        this -> getDimensions()[ tnlY ] - 1 );
         return ( Real( 1.0 ) - dx ) * this -> getElement( ix,
                                                           this -> getDimensions()[ tnlY ] - 1 ) +
                                                           dx * this -> getElement( ix + 1,
                                                           this -> getDimensions()[ tnlY ] - 1 );
      }
      if( ix >= this -> getDimensions()[ tnlX ] - 1 )
         return ( Real( 1.0 ) - dy ) * this -> getElement( this -> getDimensions()[ tnlX ] - 1,
                                                           iy ) +
                                                           dy * this -> getElement( this -> getDimensions()[ tnlX ] - 1,
                                                           iy + 1 );
      Real a1, a2;
      a1 = ( Real( 1.0 ) - dx ) * this -> getElement( ix, iy ) +
                             dx * this -> getElement( ix + 1, iy );

      a2 = ( Real( 1.0 ) - dx ) * this -> getElement( ix, iy + 1 ) +
                             dx * this -> getElement( ix + 1, iy + 1 );
      return ( Real( 1.0 ) - dy ) * a1 + dy * a2;
   }
   if( Dimensions == 3 )
   {
      tnlAssert( 0, cerr << "Interpolation is not implemented yet.");
   }
   if( Dimensions > 3 )
   {
      cerr << "Interpolation for 4D grids is not implemented yet." << endl;
      abort();
   }
}

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: getValue( const Real& x ) const
{
   return this -> getValue( tnlTuple< 1, Real >( x ) );
}

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: getValue( const Real& x,
                                                             const Real& y ) const
{
   return this -> getValue( tnlTuple< 2, Real >( x, y ) );
}

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: getValue( const Real& x,
                                                             const Real& y,
                                                             const Real& z ) const
{
   return this -> getValue( tnlTuple< 3, Real >( x, y, z ) );
}

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: Partial_x_f( const Index i1 ) const
{
   tnlAssert( Dimensions == 1,
              cerr << "The array " << this -> getName()
                   << " has " << Dimensions << " but 1 is expected." << endl; );
   tnlAssert( i1 >= 0 &&
              i1 < ( this -> getDimensions(). x() - 1 ),
              cerr << " i1 = " << i1 << " and it should be in [ 0, " <<
                   ( this -> getDimensions(). y() - 1  ) << " )" << endl; );

   const Real& Hx = spaceSteps[ 0 ];
   tnlAssert( Hx > 0, cerr << "Hx = " << Hx << endl; );
   return ( tnlMultiArray< 1, Real, tnlHost, Index > :: getElement( i1 + 1 ) -
            tnlMultiArray< 1, Real, tnlHost, Index > ::  getElement( i1 ) ) / Hx;
};

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: Partial_x_f( const Index i1,
                                                                const Index i2 ) const
{
   tnlAssert( Dimensions == 2,
              cerr << "The array " << this -> getName()
                   << " has " << Dimensions << " but 2 are expected." << endl; );
   tnlAssert( i1 >= 0 && i2 >= 0 &&
              i1 < this -> getDimensions(). x() - 1 && i2 <= this -> getDimensions(). y() - 1,
              cerr << " i1 = " << i1 << " and it should be in [ 0, " <<
                      ( this -> getDimensions() .x() - 1  ) << " ) " << endl
                   << " i2 = " << i2 << " and it should be in [ 0, " <<
                      ( this -> getDimensions(). y() - 1  ) << " ] " << endl; );

   const Real& Hx = spaceSteps[ 0 ];
   tnlAssert( Hx > 0, cerr << "Hx = " << Hx << endl; );
   return ( this -> getElement( i1 + 1, i2 ) -
            this ->  getElement( i1, i2 ) ) / Hx;
};

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: Partial_x_f( const Index i1,
                                                                const Index i2,
                                                                const Index i3 ) const
{
   tnlAssert( Dimensions == 3,
              cerr << "The array " << this -> getName()
                   << " has " << Dimensions << " but 3 are expected." << endl; );
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

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: Partial_x_b( const Index i1 ) const
{
   tnlAssert( Dimensions == 1,
              cerr << "The array " << this -> getName()
                   << " has " << Dimensions << " but 1 is expected." << endl; );
   tnlAssert( i1 > 0 &&
              i1 <= ( tnlMultiArray< 1, Real, tnlHost, Index > :: getDimensions()[ tnlX ] - 1 ),
              cerr << " i1 = " << i1 << " and it should be in ( 0, " <<
                   ( tnlMultiArray< 1, Real, tnlHost, Index > :: getDimensions()[ tnlX ] - 1  ) << " ] " << endl; );

   const Real& Hx = spaceSteps[ 0 ];
   tnlAssert( Hx > 0, cerr << "Hx = " << Hx << endl; );
   return ( tnlMultiArray< 1, Real, tnlHost, Index > :: getElement( i1 ) -
            tnlMultiArray< 1, Real, tnlHost, Index > :: getElement( i1 - 1 ) ) / Hx;
};

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: Partial_x_b( const Index i1,
                                                                const Index i2 ) const
{
   tnlAssert( Dimensions == 2,
              cerr << "The array " << this -> getName()
                   << " has " << Dimensions << " but 2 are expected." << endl; );
   tnlAssert( i1 > 0 && i2 >= 0 &&
              i1 <= this -> getDimensions()[ tnlX ] - 1 && i2 <= this -> getDimensions()[ tnlY ] - 1,
              cerr << " i1 = " << i1 << " and it should be in ( 0, " <<
                      ( this -> getDimensions()[ tnlX ] - 1  ) << " ] " << endl
                   << " i2 = " << i2 << " and it should be in [ 0, " <<
                      ( this -> getDimensions()[ tnlY ] - 1  ) << " ] " << endl; );

   const Real& Hx = spaceSteps[ 0 ];
   tnlAssert( Hx > 0, cerr << "Hx = " << Hx << endl; );
   return ( this -> getElement( i1, i2 ) -
            this -> getElement( i1 - 1, i2 ) ) / Hx;
};

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: Partial_x_b( const Index i1,
                                                                const Index i2,
                                                                const Index i3 ) const
{
   tnlAssert( Dimensions == 3,
              cerr << "The array " << this -> getName()
                   << " has " << Dimensions << " but 3 are expected." << endl; );
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

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: Partial_x( const Index i1 ) const
{
   tnlAssert( Dimensions == 1,
              cerr << "The array " << this -> getName()
                   << " has " << Dimensions << " but 1 is expected." << endl; );
   tnlAssert( i1 > 0 &&
              i1 < ( this -> getDimensions()[ tnlX ] - 1 ),
              cerr << " i1 = " << i1 << " and it should be in ( 0, " <<
                   ( this -> getDimensions()[ tnlX ] - 1  ) << " ) " << endl; );

   const Real& Hx = spaceSteps[ 0 ];
   tnlAssert( Hx > 0, cerr << "Hx = " << Hx << endl; );
   return ( tnlMultiArray< 1, Real, tnlHost, Index > :: getElement( i1 + 1 ) -
            tnlMultiArray< 1, Real, tnlHost, Index > :: getElement( i1 - 1 ) ) / ( 2.0 * Hx );
};

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: Partial_x( const Index i1,
                                                              const Index i2 ) const
{
   tnlAssert( Dimensions == 2,
              cerr << "The array " << this -> getName()
                   << " has " << Dimensions << " but 2 are expected." << endl; );
   tnlAssert( i1 > 0 && i2 >= 0 &&
              i1 < this -> getDimensions()[ tnlX ] - 1 && i2 <= this -> getDimensions()[ tnlY ] - 1,
              cerr << " i1 = " << i1 << " and it should be in ( 0, " <<
                      ( this -> getDimensions()[ tnlX ] - 1  ) << " ) " << endl
                   << " i2 = " << i2 << " and it should be in [ 0, " <<
                      ( this -> getDimensions()[ tnlY ] - 1  ) << " ] " << endl; );

   const Real& Hx = spaceSteps[ 0 ];
   tnlAssert( Hx > 0, cerr << "Hx = " << Hx << endl; );
   return ( this -> getElement( i1 + 1, i2 ) -
            this -> getElement( i1 - 1, i2 ) ) / ( 2.0 * Hx );
};

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: Partial_x( const Index i1,
                                                              const Index i2,
                                                              const Index i3 ) const
{
   tnlAssert( Dimensions == 3,
              cerr << "The array " << this -> getName()
                   << " has " << Dimensions << " but 3 are expected." << endl; );
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

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: Partial_xx( const Index i1 ) const
{
   tnlAssert( Dimensions == 1,
              cerr << "The array " << this -> getName()
                   << " has " << Dimensions << " but 1 is expected." << endl; );
   tnlAssert( i1 > 0 &&
              i1 < ( tnlMultiArray< 1, Real, tnlHost, Index > :: getDimensions()[ tnlX ] - 1 ),
              cerr << " i1 = " << i1 << " and it should be in ( 0, " <<
                   ( tnlMultiArray< 1, Real, tnlHost, Index > :: getDimensions()[ tnlX ] - 1  ) << " ) " << endl; );

   const Real& Hx = spaceSteps[ 0 ];
   tnlAssert( Hx > 0, cerr << "Hx = " << Hx << endl; );
   return ( tnlMultiArray< 1, Real, tnlHost, Index > :: getElement( i1 + 1 ) -
            2.0 * tnlMultiArray< 1, Real, tnlHost, Index > :: getElement( i1 ) +
            tnlMultiArray< 1, Real, tnlHost, Index > :: getElement( i1 - 1 ) ) / ( Hx * Hx );
};

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: Partial_xx( const Index i1,
                                                               const Index i2 ) const
{
   tnlAssert( Dimensions == 2,
              cerr << "The array " << this -> getName()
                   << " has " << Dimensions << " but 2 are expected." << endl; );
   tnlAssert( i1 > 0 && i2 >= 0 &&
              i1 < this -> getDimensions()[ tnlX ] - 1 && i2 <= this -> getDimensions()[ tnlY ] - 1,
              cerr << " i1 = " << i1 << " and it should be in ( 0, " <<
                      ( this -> getDimensions()[ tnlX ] - 1  ) << " ) " << endl
                   << " i2 = " << i2 << " and it should be in [ 0, " <<
                      ( this -> getDimensions()[ tnlY ] - 1  ) << " ] " << endl; );

   const Real& Hx = spaceSteps[ 0 ];
   tnlAssert( Hx > 0, cerr << "Hx = " << Hx << endl; );
   return ( this -> getElement( i1 + 1, i2 ) -
            2.0 * this -> getElement( i1, i2 ) +
            this -> getElement( i1 - 1, i2 ) ) / ( Hx * Hx );
};

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: Partial_xx( const Index i1,
                                                               const Index i2,
                                                               const Index i3 ) const
{
   tnlAssert( Dimensions == 3,
              cerr << "The array " << this -> getName()
                   << " has " << Dimensions << " but 3 are expected." << endl; );
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

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: Partial_y_f( const Index i1,
                                                                const Index i2 ) const
{
   tnlAssert( Dimensions == 2,
              cerr << "The array " << this -> getName()
                   << " has " << Dimensions << " but 2 are expected." << endl; );
   tnlAssert( i1 >= 0 && i2 >= 0 &&
              i1 <= this -> getDimensions()[ tnlX ] - 1 && i2 < this -> getDimensions()[ tnlY ] - 1,
              cerr << " i1 = " << i1 << " and it should be in [ 0, " <<
                      ( this -> getDimensions()[ tnlX ] - 1  ) << " ] " << endl
                   << " i2 = " << i2 << " and it should be in [ 0, " <<
                      ( this -> getDimensions()[ tnlY ] - 1  ) << " ) " << endl; );

   const Real& Hy = spaceSteps[ 1 ];
   tnlAssert( Hy > 0, cerr << "Hy = " << Hy << endl; );
   return ( this -> getElement( i1, i2 + 1 ) -
            this -> getElement( i1, i2 ) ) / Hy;
};

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: Partial_y_f( const Index i1,
                                                                const Index i2,
                                                                const Index i3 ) const
{
   tnlAssert( Dimensions == 3,
              cerr << "The array " << this -> getName()
                   << " has " << Dimensions << " but 3 are expected." << endl; );
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

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: Partial_y_b( const Index i1,
                                                                const Index i2 ) const
{
   tnlAssert( Dimensions == 2,
              cerr << "The array " << this -> getName()
                   << " has " << Dimensions << " but 2 are expected." << endl; );
   tnlAssert( i1 >= 0 && i2 > 0 &&
              i1 <= this -> getDimensions()[ tnlX ] - 1 && i2 <= this -> getDimensions()[ tnlY ] - 1,
              cerr << " i1 = " << i1 << " and it should be in [ 0, " <<
                      ( this -> getDimensions()[ tnlX ] - 1  ) << " ] " << endl
                   << " i2 = " << i2 << " and it should be in ( 0, " <<
                      ( this -> getDimensions()[ tnlY ] - 1  ) << " ] " << endl; );

   const Real& Hy = spaceSteps[ 1 ];
   tnlAssert( Hy > 0, cerr << "Hy = " << Hy << endl; );
   return ( this -> getElement( i1, i2 ) -
            this -> getElement( i1, i2 - 1 ) ) / Hy;
};

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: Partial_y_b( const Index i1,
                                                                const Index i2,
                                                                const Index i3 ) const
{
   tnlAssert( Dimensions == 3,
              cerr << "The array " << this -> getName()
                   << " has " << Dimensions << " but 3 are expected." << endl; );
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

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: Partial_y( const Index i1,
                                                              const Index i2 ) const
{
   tnlAssert( Dimensions == 2,
              cerr << "The array " << this -> getName()
                   << " has " << Dimensions << " but 2 are expected." << endl; );
   tnlAssert( i1 >= 0 && i2 > 0 &&
              i1 <= this -> getDimensions()[ tnlX ] - 1 && i2 < this -> getDimensions()[ tnlY ] - 1,
              cerr << " i1 = " << i1 << " and it should be in [ 0, " <<
                      ( this -> getDimensions()[ tnlX ] - 1  ) << " ] " << endl
                   << " i2 = " << i2 << " and it should be in ( 0, " <<
                      ( this -> getDimensions()[ tnlY ] - 1  ) << " ) " << endl; );

   const Real& Hy = spaceSteps[ 1 ];
   tnlAssert( Hy > 0, cerr << "Hy = " << Hy << endl; );
   return ( this -> getElement( i1, i2 + 1 ) -
            this -> getElement( i1, i2 - 1 ) ) / ( 2.0 * Hy );
};

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: Partial_y( const Index i1,
                                                              const Index i2,
                                                              const Index i3 ) const
{
   tnlAssert( Dimensions == 3,
              cerr << "The array " << this -> getName()
                   << " has " << Dimensions << " but 3 are expected." << endl; );
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

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: Partial_yy( const Index i1,
                                                               const Index i2 ) const
{
   tnlAssert( Dimensions == 2,
              cerr << "The array " << this -> getName()
                   << " has " << Dimensions << " but 2 are expected." << endl; );
   tnlAssert( i1 >= 0 && i2 > 0 &&
              i1 <= this -> getDimensions()[ tnlX ] - 1 && i2 < this -> getDimensions()[ tnlY ] - 1,
              cerr << " i1 = " << i1 << " and it should be in [ 0, " <<
                      ( this -> getDimensions()[ tnlX ] - 1  ) << " ] " << endl
                   << " i2 = " << i2 << " and it should be in ( 0, " <<
                      ( this -> getDimensions()[ tnlY ] - 1  ) << " ) " << endl; );

   const Real& Hy = spaceSteps[ 1 ];
   tnlAssert( Hy > 0, cerr << "Hy = " << Hy << endl; );
   return ( this -> getElement( i1, i2 + 1 ) -
            2.0 * this -> getElement( i1, i2 ) +
            this -> getElement( i1, i2 - 1 ) ) / ( Hy * Hy );
};

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: Partial_yy( const Index i1,
                                                               const Index i2,
                                                               const Index i3 ) const
{
   tnlAssert( Dimensions == 3,
              cerr << "The array " << this -> getName()
                   << " has " << Dimensions << " but 3 are expected." << endl; );
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

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: Partial_z_f( const Index i1,
                                                                const Index i2,
                                                                const Index i3 ) const
{
   tnlAssert( Dimensions == 3,
              cerr << "The array " << this -> getName()
                   << " has " << Dimensions << " but 3 are expected." << endl; );
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

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: Partial_z_b( const Index i1,
                                                                const Index i2,
                                                                const Index i3 ) const
{
   tnlAssert( Dimensions == 3,
              cerr << "The array " << this -> getName()
                   << " has " << Dimensions << " but 3 are expected." << endl; );
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

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: Partial_z( const Index i1,
                                                              const Index i2,
                                                              const Index i3 ) const
{
   tnlAssert( Dimensions == 3,
              cerr << "The array " << this -> getName()
                   << " has " << Dimensions << " but 3 are expected." << endl; );
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

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: Partial_zz( const Index i1,
                                                               const Index i2,
                                                               const Index i3 ) const
{
   tnlAssert( Dimensions == 3,
              cerr << "The array " << this -> getName()
                   << " has " << Dimensions << " but 3 are expected." << endl; );
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

template< int Dimensions, typename Real, typename Device, typename Index >
void tnlGrid< Dimensions, Real, Device, Index > :: setDirichletBC( const tnlGrid< Dimensions, Real, Device, Index >&bc,
                                                                   const tnlTuple< Dimensions, bool >& lowerBC,
                                                                   const tnlTuple< Dimensions, bool >& upperBC )
{
   if( Device :: getDevice() == tnlHostDevice )
   {
      if( Dimensions == 1 )
      {
         const Index xSize = this -> getDimensions(). x();
         if( lowerBC. x() )
            ( *this )( 0 ) = bc( 0 );
         if( upperBC. x() )
            ( *this )( xSize - 1 ) = bc( xSize - 1 );
      }
      if( Dimensions == 2 )
      {
         const Index xSize = this -> getDimensions(). x();
         const Index ySize = this -> getDimensions(). y();
         Index i, j;

         if( lowerBC. x() )
            for( i = 0; i < xSize; i ++ )
               ( *this )( i, 0 ) = bc( i, 0 );
         if( upperBC. x() )
            for( i = 0; i < xSize; i ++ )
               ( *this )( i, ySize - 1 ) = bc( i, ySize - 1 );

         if( lowerBC. y() )
            for( j = 0; j < ySize; j ++ )
               ( *this )( 0, j ) = bc( 0, j );
         if( upperBC. y() )
            for( j = 0; j < ySize; j ++ )
               ( *this )( xSize - 1, j ) = bc( xSize - 1, j );
      }
      if( Dimensions == 3 )
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
   }
   if( Device :: getDevice() == tnlCudaDevice )
   {
#ifdef HAVE_CUDA
      if( Dimensions == 2 )
      {
         const Index size = this -> getSize();
         const Index desBlockSize = 64;
         const Index gridSize = size / desBlockSize + ( size % desBlockSize != 0 );
         dim3 gridDim( 0 ), blockDim( 0 );
         gridDim. x = gridSize;
         blockDim. x = desBlockSize;
         :: setDirichletBC< 2, Real, Index ><<< gridDim, blockDim >>>( this -> getDimensions(). x(),
                                                                       this -> getDimensions(). y(),
                                                                       bc. getData(),
                                                                       this -> getData() );
      }
#endif
   }
}

template< int Dimensions, typename Real, typename Device, typename Index >
void tnlGrid< Dimensions, Real, Device, Index > :: setDirichletBC( const Real& bcValue,
                                                                   const tnlTuple< Dimensions, bool >& lowerBC,
                                                                   const tnlTuple< Dimensions, bool >& upperBC )
{
   if( Device :: getDevice() == tnlHostDevice )
   {
      if( Dimensions == 1 )
      {
         const Index xSize = this -> getDimensions(). x();
         if( lowerBC. x() )
            ( *this )( 0 ) = bcValue;
         if( upperBC. x() )
            ( *this )( xSize - 1 ) = bcValue;
      }
      if( Dimensions == 2 )
      {
         const Index xSize = this -> getDimensions(). x();
         const Index ySize = this -> getDimensions(). y();
         Index i, j;

         if( lowerBC. x() )
            for( i = 0; i < xSize; i ++ )
               ( *this )( i, 0 ) = bcValue;
         if( upperBC. x() )
            for( i = 0; i < xSize; i ++ )
               ( *this )( i, ySize - 1 ) = bcValue;

         if( lowerBC. y() )
            for( j = 0; j < ySize; j ++ )
               ( *this )( 0, j ) = bcValue;
         if( upperBC. y() )
            for( j = 0; j < ySize; j ++ )
               ( *this )( xSize - 1, j ) = bcValue;
      }
      if( Dimensions == 3 )
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
   }
   if( Device :: getDevice() == tnlCudaDevice )
   {
#ifdef HAVE_CUDA
      if( Dimensions == 2 )
      {
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
      }
#endif
   }
}

template< int Dimensions, typename Real, typename Device, typename Index >
void tnlGrid< Dimensions, Real, Device, Index > :: setNeumannBC( const tnlGrid< Dimensions, Real, Device, Index >&bc,
                                                                 const tnlTuple< Dimensions, bool >& lowerBC,
                                                                 const tnlTuple< Dimensions, bool >& upperBC )
{
   if( Device :: getDevice() == tnlHostDevice )
   {
      if( Dimensions == 1 )
      {
         const Real& hx = this -> getSpaceSteps(). x();

         const Index xSize = this -> getDimensions(). x();
         if( lowerBC. x() )
            ( *this )( 0 ) = ( *this )( 1 ) + hx * bc( 0 );
         if( upperBC. x() )
            ( *this )( xSize - 1 ) = ( *this )( xSize - 1 ) + hx * bc( xSize -1 );
      }
      if( Dimensions == 2 )
      {
         const Real& hx = this -> getSpaceSteps(). x();
         const Real& hy = this -> getSpaceSteps(). y();

         const Index xSize = this -> getDimensions(). x();
         const Index ySize = this -> getDimensions(). y();
         Index i, j;

         if( lowerBC. x() )
            for( i = 0; i < xSize; i ++ )
                ( *this )( i, 0 ) = ( *this )( i, 1 ) + hy * bc( i, 0 );
         if( upperBC. x() )
            for( i = 0; i < xSize; i ++ )
               ( *this )( i, ySize - 1 ) = ( *this )( i, ySize - 2 ) + hy * bc( i, ySize - 1 );

         if( lowerBC. y() )
            for( j = 0; j < ySize; j ++ )
               ( *this )( 0, j ) = ( *this )( 1, j ) + hx * bc( 0, j );
         if( upperBC. y() )
            for( j = 0; j < ySize; j ++ )
               ( *this )( xSize - 1, j ) = ( *this )( xSize - 2, j ) + hx * bc( xSize - 1, j );
      }
      if( Dimensions == 3 )
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
   }
   if( Device :: getDevice() == tnlCudaDevice )
   {
#ifdef HAVE_CUDA
      if( Dimensions == 2 )
      {
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
      }
#endif
   }
}

template< int Dimensions, typename Real, typename Device, typename Index >
void tnlGrid< Dimensions, Real, Device, Index > :: setNeumannBC( const Real& bcValue,
                                                                 const tnlTuple< Dimensions, bool >& lowerBC,
                                                                 const tnlTuple< Dimensions, bool >& upperBC )
{
   if( Device :: getDevice() == tnlHostDevice )
   {
      if( Dimensions == 1 )
      {
         const Real& hx = this -> getSpaceSteps(). x();

         const Index xSize = this -> getDimensions(). x();
         if( lowerBC. x() )
            ( *this )( 0 ) = ( *this )( 1 ) + hx * bcValue;
         if( upperBC. x() )
            ( *this )( xSize - 1 ) = ( *this )( xSize - 1 ) + hx * bcValue;
      }
      if( Dimensions == 2 )
      {
         const Real& hx = this -> getSpaceSteps(). x();
         const Real& hy = this -> getSpaceSteps(). y();

         const Index xSize = this -> getDimensions(). x();
         const Index ySize = this -> getDimensions(). y();
         Index i, j;

         if( lowerBC. x() )
            for( i = 0; i < xSize; i ++ )
               ( *this )( i, 0 ) = ( *this )( i, 1 ) + hy * bcValue;
         if( upperBC. x() )
            for( i = 0; i < xSize; i ++ )
               ( *this )( i, ySize - 1 ) = ( *this )( i, ySize - 2 ) + hy * bcValue;

         if( lowerBC. y() )
            for( j = 0; j < ySize; j ++ )
               ( *this )( 0, j ) = ( *this )( 1, j ) + hx * bcValue;
         if( upperBC. y() )
            for( j = 0; j < ySize; j ++ )
               ( *this )( xSize - 1, j ) = ( *this )( xSize - 2, j ) + hx * bcValue;
      }
      if( Dimensions == 3 )
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
   }
   if( Device :: getDevice() == tnlCudaDevice )
   {
#ifdef HAVE_CUDA
      if( Dimensions == 2 )
      {
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
      }
#endif
   }
}

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: getMax() const
{
   return tnlMax( * this );
}

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: getMin() const
{
   return tnlMin( * this );
}

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: getAbsMax() const
{
   return tnlAbsMax( * this );
}

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: getAbsMin() const
{
   return tnlAbsMin( * this );
}

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: getLpNorm() const
{
   Real result = tnlLpNorm( * this );
   if( Dimensions == 1 )
      return result * getSpaceSteps(). x();
   if( Dimensions == 2 )
      return result * getSpaceSteps(). x()
                    * getSpaceSteps(). y();
   if( Dimensions == 3 )
      return result * getSpaceSteps(). x()
                    * getSpaceSteps(). y()
                    * getSpaceSteps(). z();
   for( int i = 0; i < Dimensions; i ++ )
      result *= getSpaceSteps()[ i ];
   return result;
}

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: getSum() const
{
   return tnlMultiArray< Dimensions, Real, Device, Index > :: sum( * this );
}

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: getDifferenceMax( const tnlVector< Real, Device, Index >& v ) const
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

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: getDifferenceMin( const tnlVector< Real, Device, Index >& v ) const
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

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: getDifferenceAbsMax( const tnlVector< Real, Device, Index >& v ) const
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

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: getDifferenceAbsMin( const tnlVector< Real, Device, Index >& v ) const
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

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: getDifferenceLpNorm( const tnlVector< Real, Device, Index >& v, const Real& p ) const
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
   if( Dimensions == 1 )
      return result * getSpaceSteps(). x();
   if( Dimensions == 2 )
      return result * getSpaceSteps(). x()
                    * getSpaceSteps(). y();
   if( Dimensions == 3 )
      return result * getSpaceSteps(). x()
                    * getSpaceSteps(). y()
                    * getSpaceSteps(). z();
   for( int i = 0; i < Dimensions; i ++ )
      result *= getSpaceSteps()[ i ];
   return result;
}

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: getDifferenceSum( const tnlVector< Real, Device, Index >& v ) const
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

template< int Dimensions, typename Real, typename Device, typename Index >
void tnlGrid< Dimensions, Real, Device, Index > :: scalarMultiplication( const Real& alpha )
{
   return this -> scalarMultiplication( alpha );
}

template< int Dimensions, typename Real, typename Device, typename Index >
Real tnlGrid< Dimensions, Real, Device, Index > :: sdot( const tnlVector< Real, Device, Index >& v ) const
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
   return this -> sdot( v );
};

template< int Dimensions, typename Real, typename Device, typename Index >
void tnlGrid< Dimensions, Real, Device, Index > :: saxpy( const Real& alpha,
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

template< int Dimensions, typename Real, typename Device, typename Index >
void tnlGrid< Dimensions, Real, Device, Index > :: saxmy( const Real& alpha,
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


template< int Dimensions, typename Real, typename Device, typename Index >
bool tnlGrid< Dimensions, Real, Device, Index > :: save( tnlFile& file ) const
{
   if( ! tnlMultiArray< Dimensions, Real, Device, Index > :: save( file ) )
   {
      cerr << "I was not able to write the tnlMultiArray of the tnlGrid "
           << this -> getName() << endl;
      return false;
   }
   if( ! domainLowerCorner. save( file ) ||
       ! domainUpperCorner. save( file ) ||
       ! spaceSteps. save( file ) )
   {
      cerr << "I was not able to write the domain decsription of the tnlGrid "
           << this -> getName() << endl;
      return false;
   }
   return true;
}

template< int Dimensions, typename Real, typename Device, typename Index >
bool tnlGrid< Dimensions, Real, Device, Index > :: load( tnlFile& file )
{
   if( ! tnlMultiArray< Dimensions, Real, Device, Index > :: load( file ) )
   {
      cerr << "I was not able to read the tnlMultiArray of the tnlGrid "
           << this -> getName() << endl;
      return false;
   }
   if( ! domainLowerCorner. load( file ) ||
       ! domainUpperCorner. load( file ) ||
       ! spaceSteps. load( file ) )
   {
      cerr << "I was not able to read the domain description of the tnlGrid "
           << this -> getName() << endl;
      return false;
   }
   return true;
}

template< int Dimensions, typename Real, typename Device, typename Index >
bool tnlGrid< Dimensions, Real, Device, Index > :: save( const tnlString& fileName ) const
{
   return tnlObject :: save( fileName );
}

template< int Dimensions, typename Real, typename Device, typename Index >
bool tnlGrid< Dimensions, Real, Device, Index > :: load( const tnlString& fileName )
{
   return tnlObject :: load( fileName );
}

template< int Dimensions, typename Real, typename Device, typename Index >
bool tnlGrid< Dimensions, Real, Device, Index > :: draw( const tnlString& fileName,
                                                         const tnlString& format,
                                                         const tnlTuple< Dimensions, Index > steps ) const
{
   tnlAssert( steps > ( tnlTuple< Dimensions, Index >( 0 ) ),
              cerr << "Wrong steps of increment ( " << steps << " )"
                   << " for drawing the tnlGrid " << this -> getName() << "." << endl; );
   if( format == "tnl" )
      return this -> save( fileName );
   fstream file;
   file. open( fileName. getString(), ios :: out );
   if( ! file )
   {
      cerr << " I am not able to open the file " << fileName
           << " for drawing the tnlGrid " << this -> getName() << "." << endl;
      return false;
   }
   if( Dimensions == 1 )
   {
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
   }
   if( Dimensions == 2 )
   {
      const Index xSize = this -> getDimensions()[ tnlX ];
      const Index ySize = this -> getDimensions()[ tnlY ];
      const Real& ax = this -> getDomainLowerCorner()[ tnlX ];
      const Real& ay = this -> getDomainLowerCorner()[ tnlY ];
      const Real& hx = this -> getSpaceSteps()[ tnlX ];
      const Real& hy = this -> getSpaceSteps()[ tnlY ];
      if( format == "gnuplot" )
      {
         for( Index i = 0; i < xSize; i += steps[ tnlX ] )
         {
            for( Index j = 0; j < ySize; j += steps[ tnlY ] )
            {
               file << setprecision( 12 )
                    << ax + Real( i ) * hx * steps[ tnlX ]
                    << " "
                    << ay + Real( j ) * hy * steps[ tnlY ]
                    << " "
                    << this -> getElement( i, j )
                    << endl;
            }
            file << endl;
         }
         return true;
      }
      if( format == "vti" )
      {
         file << "<VTKFile type=\"ImagegetString\" version=\"0.1\" byte_order=\"LittleEndian\">" << endl;
         file << "<ImagegetString WholeExtent=\""
              << 0 << " " << xSize - 1 << " " << 0 << " " << ySize - 1
              << " 0 0\" Origin=\"0 0 0\" Spacing=\""
              << hx * steps[ tnlX ] << " " << hy * steps[ tnlY ] << " 0\">" << endl;
         file << "<Piece Extent=\"0 " << xSize - 1 << " 0 " << ySize - 1 <<" 0 0\">" << endl;
         file << "<PointgetString Scalars=\"order_parameter\">" << endl;
         file << "<getStringArray Name=\"order_parameter\" type=\"Float32\" format=\"ascii\">" << endl;
         file. flags( ios_base::scientific );
         Index iStep = steps[ tnlX ];
         Index jStep = steps[ tnlY ];
         for( Index j = 0; j <= ySize - jStep; j += jStep )
            for( Index i = 0; i <= xSize - iStep; i += iStep )
                 file << this -> getElement( i, j ) << " ";
         file << endl;
         file << "</getStringArray>" << endl;
         file << "</PointgetString>" << endl;
         file << "</Piece>" << endl;
         file << "</ImagegetString>" << endl;
         file << "</VTKFile>" << endl;
         return true;
      }
   }
   if( Dimensions == 3 )
   {
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
         cout << "GNUPLOT is not supported for tnlGrid3D." << endl;
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

   }
   cerr << endl << "I do not know a format " << format << " for tnlGrid with " << Dimensions << " dimensions.";
   return false;
}

#ifdef HAVE_CUDA
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

#endif


#endif /* TNLGRID_H_ */
