/***************************************************************************
                          tnlGrid2D.h  -  description
                             -------------------
    begin                : 2007/06/17
    copyright            : (C) 2007 by Tomas Oberhuber
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

#ifndef tnlGrid2DH
#define tnlGrid2DH

#include <core/tnlField2D.h>

template< class T > class tnlGridCUDA2D;

template< class T = double > class tnlGrid2D :
	                         public tnlField2D< T >
{
   public:

   tnlGrid2D( const char* name = 0 )
   : tnlField2D< T >( name )
   {};

   //! Constructor with the grid and the domain dimensions
   /*! @param x_size and @param y_size define the grid dimensions.
       @param A_x, @param B_x, @param A_y and @param B_y define domain
       Omega = <A_x,B_x>*<A_y,B_y>.
    */
   tnlGrid2D( const char* name,
              int x_size,
              int y_size,
              const double& A_x,
              const double& B_x,
              const double& A_y,
              const double& B_y )
   : tnlField2D< T >( name, x_size, y_size ),
     Ax( A_x ), Bx( B_x ),
     Ay( A_y ), By( B_y )  
   {
      assert( Ax < Bx && Ay < By );
      Hx = ( Bx - Ax ) / ( double ) ( tnlField2D< T > :: GetXSize() - 1 ); 
      Hy = ( By - Ay ) / ( double ) ( tnlField2D< T > :: GetYSize() - 1 );
   };

   tnlGrid2D( const tnlGrid2D& g )
   : tnlField2D< T >( g ),
     Ax( g. Ax ), Bx( g. Bx ),
     Ay( g. Ay ), By( g. By ),
     Hx( g. Hx ), Hy( g. Hy )
   {
   };

   tnlGrid2D( const tnlGridCUDA2D< T >& g );

   tnlString GetType() const
   {
      T t;
      return tnlString( "tnlGrid2D< " ) + tnlString( GetParameterType( t ) ) + tnlString( " >" );
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
      if( ! hx ) Hx = ( Bx - Ax ) / ( double ) ( tnlField2D< T > :: GetXSize() - 1 ); 
      else Hx = hx;
      if( ! hy ) Hy = ( By - Ay ) / ( double ) ( tnlField2D< T > :: GetYSize() - 1 );
      else Hy = hy;
   }
   
   void SetNewDomain( const tnlGrid2D< T >& u )
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
      if( iy >= tnlField2D< T > :: y_size - 1 )
      {
         if( ix >= tnlField2D< T > :: x_size - 1 )
            return  tnlField2D< T > :: operator()( tnlField2D< T > :: x_size - 1, 
                                                 tnlField2D< T > :: y_size - 1 );
         return ( 1.0 - dx ) * tnlField2D< T > :: operator()( ix,
                                                            tnlField2D< T > :: y_size - 1 ) +
                          dx * tnlField2D< T > :: operator()( ix + 1, 
                                                            tnlField2D< T > :: y_size - 1 );
      }
      if( ix >= tnlField2D< T > :: x_size - 1 )
         return ( 1.0 - dy ) * tnlField2D< T > :: operator()( tnlField2D< T > :: x_size - 1,
                                                            iy ) +
                          dy * tnlField2D< T > :: operator()( tnlField2D< T > :: x_size - 1,
                                                            iy + 1 );
      N a1, a2;
      a1 = ( 1.0 - dx ) * tnlField2D< T > :: operator()( ix, iy ) +
                     dx * tnlField2D< T > :: operator()( ix + 1, iy );

      a2 = ( 1.0 - dx ) * tnlField2D< T > :: operator()( ix, iy + 1 ) +
                     dx * tnlField2D< T > :: operator()( ix + 1, iy + 1 );
      return ( 1.0 - dy ) * a1 + dy * a2;
   }
   
   //! Forward difference w.r.t x
   T Partial_x_f( const int i,
                  const int j ) const
   {
      assert( i >= 0 && j >= 0 );
      assert( i < tnlField2D< T > :: x_size - 1 && j < tnlField2D< T > :: y_size );
      return ( tnlField2D< T > :: operator()( i + 1, j ) - 
               tnlField2D< T > ::  operator()( i, j ) ) / Hx;
   };
   
   //! Backward difference w.r.t x
   T Partial_x_b( const int i,
                  const int j ) const
   {
      assert( i > 0 && j >= 0 );
      assert( i < tnlField2D< T > :: x_size && j < tnlField2D< T > :: y_size );
      return ( tnlField2D< T > :: operator()( i, j ) - 
               tnlField2D< T > :: operator()( i - 1, j ) ) / Hx;
   };

   //! Forward difference w.r.t y
   T Partial_y_f( const int i,
                  const int j ) const
   {
      assert( i >= 0 && j >= 0 );
      assert( i < tnlField2D< T > :: x_size && 
              j < tnlField2D< T > :: y_size - 1 );
      return ( tnlField2D< T > :: operator()( i, j + 1 ) - 
               tnlField2D< T > :: operator()( i, j ) ) / Hy;
   };
   
   //! Backward difference w.r.t y
   T Partial_y_b( const int i,
                  const int j ) const
   {
      assert( i >= 0 && j > 0 );
      assert( i < tnlField2D< T > :: x_size && 
              j < tnlField2D< T > :: y_size );
      return ( tnlField2D< T > :: operator()( i, j ) - 
               tnlField2D< T > :: operator()( i, j - 1 ) ) / Hy;
   };
   
   //! Central difference w.r.t. x
   T Partial_x( const int i,
                const int j ) const
   {
      assert( i > 0 && j >= 0 );
      assert( i < tnlField2D< T > :: x_size - 1 && 
              j < tnlField2D< T > :: y_size );
      return ( tnlField2D< T > :: operator()( i + 1, j ) - 
               tnlField2D< T > :: operator()( i - 1, j ) ) / ( 2.0 * Hx );
   };

   //! Central difference w.r.t y
   T Partial_y( const int i,
                const int j ) const
   {
      assert( i >= 0 && j > 0 );
      assert( i < tnlField2D< T > :: x_size && 
              j < tnlField2D< T > :: y_size - 1 );
      return ( tnlField2D< T > :: operator()( i, j + 1 ) - 
               tnlField2D< T > :: operator()( i, j - 1 ) ) / ( 2.0 * Hy );
   };
   
   //! Second order difference w.r.t. x
   T Partial_xx( const int i,
                 const int j ) const
   {
      assert( i > 0 && j >= 0 );
      assert( i < tnlField2D< T > :: x_size - 1 && 
              j < tnlField2D< T > :: y_size );
      return ( tnlField2D< T > :: operator()( i + 1, j ) - 
               2.0 * tnlField2D< T > :: operator()( i, j ) + 
               tnlField2D< T > :: operator()( i - 1, j ) ) / ( Hx * Hx );   
   };
   
   //! Second order difference w.r.t. y
   T Partial_yy( const int i,
                 const int j ) const
   {
      assert( i >= 0 && j > 0 );
      assert( i < tnlField2D< T > :: x_size && 
              j < tnlField2D< T > :: y_size - 1 );
      return ( tnlField2D< T > :: operator()( i, j + 1 ) - 
               2.0 * tnlField2D< T > :: operator()( i, j ) + 
               tnlField2D< T > :: operator()( i, j - 1 ) ) / ( Hy * Hy ); 
   };

   //! Method for saving the object to a file as a binary data
   bool Save( ostream& file ) const
   {
      if( ! tnlField2D< T > :: Save( file ) ) return false;
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
      if( ! tnlField2D< T > :: Load( file ) ) return false;
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

#include <diff/tnlGridCUDA2D.h>

template< class T > tnlGrid2D< T > :: tnlGrid2D( const tnlGridCUDA2D< T >& g )
#ifdef HAVE_CUDA
: tnlField2D< T >( g ),
         Ax( g. GetAx() ), Bx( g. GetBx() ),
         Ay( g. GetAy() ), By( g. GetBy() ),
         Hx( g. GetHx() ), Hy( g. GetHy() )
 {
 };
#else
{
   cerr << "CUDA is not supported on this system " << __FILE__ << " line " << __LINE__ << endl;
};
#endif

// Explicit instatiation
template class tnlGrid2D< double >;

#endif
