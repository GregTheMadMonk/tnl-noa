/***************************************************************************
                          mGrid3D.h  -  description
                             -------------------
    begin                : 2009/07/21
    copyright            : (C) 2009 by Tomas Oberhuber
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

#ifndef mGrid3DH
#define mGrid3DH

#include <core/mField3D.h>

template< typename T = double > class mGrid3D : public mField3D< T >
{
   public:

   mGrid3D()
   {
   };

   //! Constructor with the grid and the domain dimensions
   /*! @param x_size, @param y_size and @param z_size define the grid dimensions.
       @param A_x, @param B_x, @param A_y, @param B_y, @param A_z and @param B_zdefine domain
       Omega = <A_x,B_x>*<A_y,B_y>*,A_z,B_z>.
    */
   mGrid3D( int x_size,
            int y_size,
            int z_size,
            const double& A_x,
            const double& B_x,
            const double& A_y,
            const double& B_y,
            const double& A_z,
            const double& B_z )
   : mField3D< T >( x_size, y_size, z_size ),
     Ax( A_x ), Bx( B_x ),
     Ay( A_y ), By( B_y ),  
     Az( A_z ), Bz( B_z )  
   {
      assert( Ax < Bx && Ay < By && Az < Bz );
      Hx = ( Bx - Ax ) / ( double ) ( x_size - 1 ); 
      Hy = ( By - Ay ) / ( double ) ( y_size - 1 );
      Hz = ( Bz - Az ) / ( double ) ( z_size - 1 );
   };

   mGrid3D( const mGrid3D& g )
   : mField3D< T >( g ),
     Ax( g. Ax ), Bx( g. Bx ),
     Ay( g. Ay ), By( g. By ),
     Az( g. Ay ), Bz( g. By ),
     Hx( g. Hx ), Hy( g. Hy ), Hz( g. Hz )
   {
   };

   tnlString GetType() const
   {
      T t;
      return tnlString( "mGrid3D< " ) + tnlString( GetParameterType( t ) ) + tnlString( " >" );
   };

   void SetNewDomain( const double& A_x,
                      const double& B_x,
                      const double& A_y,
                      const double& B_y,
                      const double& A_z,
                      const double& B_z,
                      double hx = 0.0,
                      double hy = 0.0,
                      double hz = 0.0 )
   {
      Ax = A_x;
      Bx = B_x;
      Ay = A_y;
      By = B_y;
      Az = A_z;
      Bz = B_z;
      assert( Ax < Bx && Ay < By && Az < Bz );
      if( ! hx ) Hx = ( Bx - Ax ) / ( double ) ( mField3D< T > :: GetXSize() - 1 ); 
      else Hx = hx;
      if( ! hy ) Hy = ( By - Ay ) / ( double ) ( mField3D< T > :: GetYSize() - 1 );
      else Hy = hy;
      if( ! hz ) Hz = ( Bz - Az ) / ( double ) ( mField3D< T > :: GetZSize() - 1 );
      else Hz = hz;
   }
   
   void SetNewDomain( const mGrid3D< T >& u )
   {
      SetNewDomain( u. GetAx(), u. GetBx(),
                    u. GetAy(), u. GetBy(),
                    u. GetAz(), u. GetBz(),
                    u. GetHx(), u. GetHy(), u. GetHz() );
   }

   const double& GetAx() const
   {
      return Ax;
   }

   const double& GetAy() const
   {
      return Ay;
   }

   const double& GetAz() const
   {
      return Az;
   }

   const double& GetBx() const
   {
      return Bx;
   }

   const double& GetBy() const
   {
      return By;
   }

   const double& GetBz() const
   {
      return Bz;
   }
   
   const double& GetHx() const
   {
      return Hx;
   }

   const double& GetHy() const
   {
      return Hy;
   }

   const double& GetHz() const
   {
      return Hz;
   }
   
   // Interpolation
   template< typename N > T Value( N x, N y, N z ) const
   {
      x = ( x - Ax ) / Hx;
      y = ( y - Ay ) / Hy;
      z = ( z - Az ) / Hz;
      int ix = ( int ) ( x );
      int iy = ( int ) ( y );
      int iz = ( int ) ( z );
      N dx = x - ( N ) ix;
      N dy = y - ( N ) iy;
      N dz = z - ( N ) iz;

      //------------- FINISH IT FROM HERE ---------------------
      assert( 0 );
      /*if( iy >= tnlField2D< T > :: y_size - 1 )
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
      */
   }
   
   //! Forward difference w.r.t x
   T Partial_x_f( const int i,
                  const int j,
                  const int k ) const
   {
      assert( i >= 0 );
      assert( j >= 0 );
      assert( k >= 0 );
      assert( i < mField3D< T > :: x_size - 1 );
      assert( j < mField3D< T > :: y_size );
      assert( k < mField3D< T > :: z_size );
      return ( mField3D< T > :: operator()( i + 1, j, k ) - 
               mField3D< T > ::  operator()( i, j, k ) ) / Hx;
   };
   
   //! Backward difference w.r.t x
   T Partial_x_b( const int i,
                  const int j,
                  const int k ) const
   {
      assert( i > 0 );
      assert( j >= 0 );
      assert( k >= 0 );
      assert( i < mField3D< T > :: x_size );
      assert( j < mField3D< T > :: y_size );
      assert( k < mField3D< T > :: z_size );
      return ( mField3D< T > :: operator()( i, j, k ) - 
               mField3D< T > ::  operator()( i - 1, j, k ) ) / Hx;
   };

   //! Forward difference w.r.t y
   T Partial_y_f( const int i,
                  const int j,
                  const int k ) const
   {
      assert( i >= 0 );
      assert( j >= 0 );
      assert( k >= 0 );
      assert( i < mField3D< T > :: x_size );
      assert( j < mField3D< T > :: y_size - 1 );
      assert( k < mField3D< T > :: z_size );
      return ( mField3D< T > :: operator()( i, j + 1, k ) - 
               mField3D< T > :: operator()( i, j, k ) ) / Hy;
   };
   
   //! Backward difference w.r.t y
   T Partial_y_b( const int i,
                  const int j,
                  const int k ) const
   {
      assert( i >= 0 );
      assert( j > 0 );
      assert( k >= 0 );
      assert( i < mField3D< T > :: x_size );
      assert( j < mField3D< T > :: y_size );
      assert( k < mField3D< T > :: z_size );

      return ( mField3D< T > :: operator()( i, j, k ) - 
               mField3D< T > :: operator()( i, j - 1, k ) ) / Hy;
   };

   //! Forward difference w.r.t z
   T Partial_z_f( const int i,
                  const int j,
                  const int k ) const
   {
      assert( i >= 0 );
      assert( j >= 0 );
      assert( k >= 0 );
      assert( i < mField3D< T > :: x_size );
      assert( j < mField3D< T > :: y_size );
      assert( k < mField3D< T > :: z_size - 1 );
      return ( mField3D< T > :: operator()( i, j, k + 1 ) - 
               mField3D< T > :: operator()( i, j, k ) ) / Hz;
   };
   
   //! Backward difference w.r.t z
   T Partial_z_b( const int i,
                  const int j,
                  const int k ) const
   {
      assert( i >= 0 );
      assert( j >= 0 );
      assert( k > 0 );
      assert( i < mField3D< T > :: x_size );
      assert( j < mField3D< T > :: y_size );
      assert( k < mField3D< T > :: z_size );

      return ( mField3D< T > :: operator()( i, j, k ) - 
               mField3D< T > :: operator()( i, j, k - 1 ) ) / Hz;
   };
   
   //! Central difference w.r.t. x
   T Partial_x( const int i,
                const int j,
                const int k ) const
   {
      assert( i > 0 );
      assert( j >= 0 );
      assert( k >= 0 );
      assert( i < mField3D< T > :: x_size - 1 );
      assert( j < mField3D< T > :: y_size );
      assert( k < mField3D< T > :: z_size );

      return ( mField3D< T > :: operator()( i + 1, j, k ) - 
               mField3D< T > :: operator()( i - 1, j, k ) ) / ( 2.0 * Hx );
   };

   //! Central difference w.r.t y
   T Partial_y( const int i,
                const int j,
                const int k ) const
   {
      assert( i >= 0 );
      assert( j > 0 );
      assert( k >= 0 );
      assert( i < mField3D< T > :: x_size );
      assert( j < mField3D< T > :: y_size - 1 );
      assert( k < mField3D< T > :: z_size );
      return ( mField3D< T > :: operator()( i, j + 1, k ) - 
               mField3D< T > :: operator()( i, j - 1, k ) ) / ( 2.0 * Hy );
   };

   //! Central difference w.r.t z
   T Partial_z( const int i,
                const int j,
                const int k ) const
   {
      assert( i >= 0 );
      assert( j >= 0 );
      assert( k > 0 );
      assert( i < mField3D< T > :: x_size );
      assert( j < mField3D< T > :: y_size  );
      assert( k < mField3D< T > :: z_size - 1 );
      return ( mField3D< T > :: operator()( i, j, k + 1 ) - 
               mField3D< T > :: operator()( i, j, k - 1 ) ) / ( 2.0 * Hz );
   };
   
   //! Second order difference w.r.t. x
   T Partial_xx( const int i,
                 const int j,
                 const int k ) const
   {
      assert( i > 0 );
      assert( j >= 0 );
      assert( k >= 0 );
      assert( i < mField3D< T > :: x_size - 1 );
      assert( j < mField3D< T > :: y_size  );
      assert( k < mField3D< T > :: z_size );

      return ( mField3D< T > :: operator()( i + 1, j, k ) - 
               2.0 * mField3D< T > :: operator()( i, j, k ) + 
               mField3D< T > :: operator()( i - 1, j, k ) ) / ( Hx * Hx );   
   };
   
   //! Second order difference w.r.t. y
   T Partial_yy( const int i,
                 const int j,
                 const int k ) const
   {
      assert( i >= 0 );
      assert( j > 0 );
      assert( k >= 0 );
      assert( i < mField3D< T > :: x_size );
      assert( j < mField3D< T > :: y_size - 1 );
      assert( k < mField3D< T > :: z_size );

      return ( mField3D< T > :: operator()( i, j + 1, k ) - 
               2.0 * mField3D< T > :: operator()( i, j, k ) + 
               mField3D< T > :: operator()( i, j - 1, k ) ) / ( Hy * Hy ); 
   };
   
   //! Second order difference w.r.t. z
   T Partial_zz( const int i,
                 const int j,
                 const int k ) const
   {
      assert( i >= 0 );
      assert( j >= 0 );
      assert( k > 0 );
      assert( i < mField3D< T > :: x_size );
      assert( j < mField3D< T > :: y_size );
      assert( k < mField3D< T > :: z_size - 1 );

      return ( mField3D< T > :: operator()( i, j, k + 1 ) - 
               2.0 * mField3D< T > :: operator()( i, j, k ) + 
               mField3D< T > :: operator()( i, j, k - 1 ) ) / ( Hz * Hz ); 
   };

   //! Method for saving the object to a file as a binary data
   bool Save( ostream& file ) const
   {
      if( ! mField3D< T > :: Save( file ) ) return false;
      file. write( ( char* ) &Ax, sizeof( double ) );
      file. write( ( char* ) &Ay, sizeof( double ) );
      file. write( ( char* ) &Az, sizeof( double ) );
      file. write( ( char* ) &Bx, sizeof( double ) );
      file. write( ( char* ) &By, sizeof( double ) );
      file. write( ( char* ) &Bz, sizeof( double ) );
      file. write( ( char* ) &Hx, sizeof( double ) );
      file. write( ( char* ) &Hy, sizeof( double ) );
      file. write( ( char* ) &Hz, sizeof( double ) );
      if( file. bad() ) return false;
      return true;
   };

   //! Method for restoring the object from a file
   bool Load( istream& file )
   {
      if( ! mField3D< T > :: Load( file ) ) return false;
      file. read( ( char* ) &Ax, sizeof( double ) );
      file. read( ( char* ) &Ay, sizeof( double ) );
      file. read( ( char* ) &Az, sizeof( double ) );
      file. read( ( char* ) &Bx, sizeof( double ) );
      file. read( ( char* ) &By, sizeof( double ) );
      file. read( ( char* ) &Bz, sizeof( double ) );
      file. read( ( char* ) &Hx, sizeof( double ) );
      file. read( ( char* ) &Hy, sizeof( double ) );
      file. read( ( char* ) &Hz, sizeof( double ) );
      if( file. bad() ) return false;
      return true;
   };   

   protected:

   //! Domain dimensions
   double Ax, Bx, Ay, By, Az, Bz;

   //! The grid space steps
   double Hx, Hy, Hz;
};

// Explicit instatiation
template class mGrid3D< double >;

#endif
