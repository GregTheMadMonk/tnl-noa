/***************************************************************************
                          mGrid2D.h  -  description
                             -------------------
    begin                : 2007/06/17
    copyright            : (C) 2007 by Tomá¹ Oberhuber
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

#ifndef mGrid2DH
#define mGrid2DH

#include <core/mField2D.h>

template< class T = double > class mGrid2D :
	                         public mField2D< T >
{
   public:

   mGrid2D()
   {
   };

   //! Constructor with the grid and the domain dimensions
   /*! @param x_size and @param y_size define the grid dimensions.
       @param A_x, @param B_x, @param A_y and @param B_y define domain
       Omega = <A_x,B_x>*<A_y,B_y>.
    */
   mGrid2D( int x_size,
            int y_size,
            const double& A_x,
            const double& B_x,
            const double& A_y,
            const double& B_y )
   : mField2D< T >( x_size, y_size ),
     Ax( A_x ), Bx( B_x ),
     Ay( A_y ), By( B_y )  
   {
      assert( Ax < Bx && Ay < By );
      Hx = ( Bx - Ax ) / ( double ) ( mField2D< T > :: GetXSize() - 1 ); 
      Hy = ( By - Ay ) / ( double ) ( mField2D< T > :: GetYSize() - 1 );
   };

   mGrid2D( const mGrid2D& g )
   : mField2D< T >( g ),
     Ax( g. Ax ), Bx( g. Bx ),
     Ay( g. Ay ), By( g. By ),
     Hx( g. Hx ), Hy( g. Hy )
   {
   };

   tnlString GetType() const
   {
      T t;
      return tnlString( "mGrid2D< " ) + tnlString( GetParameterType( t ) ) + tnlString( " >" );
   };

   void SetNewDomain( const double& A_x,
                      const double& B_x,
                      const double& A_y,
                      const double& B_y,
                      double hx = 0.0,
                      double hy = 0.0 )
   {
      Ax = A_x;
      Bx = B_x;
      Ay = A_y;
      By = B_y;
      assert( Ax < Bx && Ay < By );
      if( ! hx ) Hx = ( Bx - Ax ) / ( double ) ( mField2D< T > :: GetXSize() - 1 ); 
      else Hx = hx;
      if( ! hy ) Hy = ( By - Ay ) / ( double ) ( mField2D< T > :: GetYSize() - 1 );
      else Hy = hy;
   }
   
   void SetNewDomain( const mGrid2D< T >& u )
   {
      SetNewDomain( u. GetAx(),
                    u. GetBx(),
                    u. GetAy(),
                    u. GetBy() );
   }

   const double& GetAx() const
   {
      return Ax;
   }

   const double& GetAy() const
   {
      return Ay;
   }

   const double& GetBx() const
   {
      return Bx;
   }

   const double& GetBy() const
   {
      return By;
   }
   
   const double& GetHx() const
   {
      return Hx;
   }

   const double& GetHy() const
   {
      return Hy;
   }
   
   // Interpolation
   template< typename N > T Value( N x, N y ) const
   {
      x = ( x - Ax ) / Hx;
      y = ( y - Ay ) / Hy;
      int ix = ( int ) ( x );
      int iy = ( int ) ( y );
      N dx = x - ( N ) ix;
      N dy = y - ( N ) iy;
      if( iy >= mField2D< T > :: y_size - 1 )
      {
         if( ix >= mField2D< T > :: x_size - 1 )
            return  mField2D< T > :: operator()( mField2D< T > :: x_size - 1, 
                                                 mField2D< T > :: y_size - 1 );
         return ( 1.0 - dx ) * mField2D< T > :: operator()( ix,
                                                            mField2D< T > :: y_size - 1 ) +
                          dx * mField2D< T > :: operator()( ix + 1, 
                                                            mField2D< T > :: y_size - 1 );
      }
      if( ix >= mField2D< T > :: x_size - 1 )
         return ( 1.0 - dy ) * mField2D< T > :: operator()( mField2D< T > :: x_size - 1,
                                                            iy ) +
                          dy * mField2D< T > :: operator()( mField2D< T > :: x_size - 1,
                                                            iy + 1 );
      N a1, a2;
      a1 = ( 1.0 - dx ) * mField2D< T > :: operator()( ix, iy ) +
                     dx * mField2D< T > :: operator()( ix + 1, iy );

      a2 = ( 1.0 - dx ) * mField2D< T > :: operator()( ix, iy + 1 ) +
                     dx * mField2D< T > :: operator()( ix + 1, iy + 1 );
      return ( 1.0 - dy ) * a1 + dy * a2;
   }
   
   //! Forward difference w.r.t x
   T Partial_x_f( const int i,
                  const int j ) const
   {
      assert( i >= 0 && j >= 0 );
      assert( i < mField2D< T > :: x_size - 1 && j < mField2D< T > :: y_size );
      return ( mField2D< T > :: operator()( i + 1, j ) - 
               mField2D< T > ::  operator()( i, j ) ) / Hx;
   };
   
   //! Backward difference w.r.t x
   T Partial_x_b( const int i,
                  const int j ) const
   {
      assert( i > 0 && j >= 0 );
      assert( i < mField2D< T > :: x_size && j < mField2D< T > :: y_size );
      return ( mField2D< T > :: operator()( i, j ) - 
               mField2D< T > :: operator()( i - 1, j ) ) / Hx;
   };

   //! Forward difference w.r.t y
   T Partial_y_f( const int i,
                  const int j ) const
   {
      assert( i >= 0 && j >= 0 );
      assert( i < mField2D< T > :: x_size && 
              j < mField2D< T > :: y_size - 1 );
      return ( mField2D< T > :: operator()( i, j + 1 ) - 
               mField2D< T > :: operator()( i, j ) ) / Hy;
   };
   
   //! Backward difference w.r.t y
   T Partial_y_b( const int i,
                  const int j ) const
   {
      assert( i >= 0 && j > 0 );
      assert( i < mField2D< T > :: x_size && 
              j < mField2D< T > :: y_size );
      return ( mField2D< T > :: operator()( i, j ) - 
               mField2D< T > :: operator()( i, j - 1 ) ) / Hy;
   };
   
   //! Central difference w.r.t. x
   T Partial_x( const int i,
                const int j ) const
   {
      assert( i > 0 && j >= 0 );
      assert( i < mField2D< T > :: x_size - 1 && 
              j < mField2D< T > :: y_size );
      return ( mField2D< T > :: operator()( i + 1, j ) - 
               mField2D< T > :: operator()( i - 1, j ) ) / ( 2.0 * Hx );
   };

   //! Central difference w.r.t y
   T Partial_y( const int i,
                const int j ) const
   {
      assert( i >= 0 && j > 0 );
      assert( i < mField2D< T > :: x_size && 
              j < mField2D< T > :: y_size - 1 );
      return ( mField2D< T > :: operator()( i, j + 1 ) - 
               mField2D< T > :: operator()( i, j - 1 ) ) / ( 2.0 * Hy );
   };
   
   //! Second order difference w.r.t. x
   T Partial_xx( const int i,
                 const int j ) const
   {
      assert( i > 0 && j >= 0 );
      assert( i < mField2D< T > :: x_size - 1 && 
              j < mField2D< T > :: y_size );
      return ( mField2D< T > :: operator()( i + 1, j ) - 
               2.0 * mField2D< T > :: operator()( i, j ) + 
               mField2D< T > :: operator()( i - 1, j ) ) / ( Hx * Hx );   
   };
   
   //! Second order difference w.r.t. y
   T Partial_yy( const int i,
                 const int j ) const
   {
      assert( i >= 0 && j > 0 );
      assert( i < mField2D< T > :: x_size && 
              j < mField2D< T > :: y_size - 1 );
      return ( mField2D< T > :: operator()( i, j + 1 ) - 
               2.0 * mField2D< T > :: operator()( i, j ) + 
               mField2D< T > :: operator()( i, j - 1 ) ) / ( Hy * Hy ); 
   };

   //! Method for saving the object to a file as a binary data
   bool Save( ostream& file ) const
   {
      if( ! mField2D< T > :: Save( file ) ) return false;
      file. write( ( char* ) &Ax, sizeof( double ) );
      file. write( ( char* ) &Ay, sizeof( double ) );
      file. write( ( char* ) &Bx, sizeof( double ) );
      file. write( ( char* ) &By, sizeof( double ) );
      file. write( ( char* ) &Hx, sizeof( double ) );
      file. write( ( char* ) &Hy, sizeof( double ) );
      if( file. bad() ) return false;
      return true;
   };

   //! Method for restoring the object from a file
   bool Load( istream& file )
   {
      if( ! mField2D< T > :: Load( file ) ) return false;
      file. read( ( char* ) &Ax, sizeof( double ) );
      file. read( ( char* ) &Ay, sizeof( double ) );
      file. read( ( char* ) &Bx, sizeof( double ) );
      file. read( ( char* ) &By, sizeof( double ) );
      file. read( ( char* ) &Hx, sizeof( double ) );
      file. read( ( char* ) &Hy, sizeof( double ) );
      if( file. bad() ) return false;
      return true;
   };   

   protected:

   //! Domain dimensions
   double Ax, Bx, Ay, By;

   //! The grid space steps
   double Hx, Hy;
};

// Explicit instatiation
template class mGrid2D< double >;

#endif
