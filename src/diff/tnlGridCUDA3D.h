/***************************************************************************
                          tnlGridCUDA3D.h  -  description
                             -------------------
    begin                : 2010/01/12
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

#ifndef tnlGridCUDA3DH
#define tnlGridCUDA3DH

#include <core/tnlFieldCUDA3D.h>

template< typename T = double > class tnlGridCUDA3D : public tnlFieldCUDA3D< T >
{
   public:

   tnlGridCUDA3D( const char* name = 0 )
   : tnlFieldCUDA3D( name )
   {
   };

   //! Constructor with the grid and the domain dimensions
   /*! @param x_size, @param y_size and @param z_size define the grid dimensions.
       @param A_x, @param B_x, @param A_y, @param B_y, @param A_z and @param B_zdefine domain
       Omega = <A_x,B_x>*<A_y,B_y>*,A_z,B_z>.
    */
   tnlGridCUDA3D( int x_size,
            int y_size,
            int z_size,
            const double& A_x,
            const double& B_x,
            const double& A_y,
            const double& B_y,
            const double& A_z,
            const double& B_z )
   : tnlFieldCUDA3D< T >( x_size, y_size, z_size ),
     Ax( A_x ), Bx( B_x ),
     Ay( A_y ), By( B_y ),  
     Az( A_z ), Bz( B_z )  
   {
      assert( Ax < Bx && Ay < By && Az < Bz );
      Hx = ( Bx - Ax ) / ( double ) ( x_size - 1 ); 
      Hy = ( By - Ay ) / ( double ) ( y_size - 1 );
      Hz = ( Bz - Az ) / ( double ) ( z_size - 1 );
   };

   tnlGridCUDA3D( const tnlGridCUDA3D& g )
   : tnlFieldCUDA3D< T >( g ),
     Ax( g. Ax ), Bx( g. Bx ),
     Ay( g. Ay ), By( g. By ),
     Az( g. Ay ), Bz( g. By ),
     Hx( g. Hx ), Hy( g. Hy ), Hz( g. Hz )
   {
   };

   tnlString GetType() const
   {
      T t;
      return tnlString( "tnlGridCUDA3D< " ) + tnlString( GetParameterType( t ) ) + tnlString( " >" );
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
      if( ! hx ) Hx = ( Bx - Ax ) / ( double ) ( tnlFieldCUDA3D< T > :: GetXSize() - 1 );
      else Hx = hx;
      if( ! hy ) Hy = ( By - Ay ) / ( double ) ( tnlFieldCUDA3D< T > :: GetYSize() - 1 );
      else Hy = hy;
      if( ! hz ) Hz = ( Bz - Az ) / ( double ) ( tnlFieldCUDA3D< T > :: GetZSize() - 1 );
      else Hz = hz;
   }
   
   void SetNewDomain( const tnlGridCUDA3D< T >& u )
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
   
   protected:

   //! Domain dimensions
   double Ax, Bx, Ay, By, Az, Bz;

   //! The grid space steps
   double Hx, Hy, Hz;
};

// Explicit instatiation
template class tnlGridCUDA3D< double >;

#endif
