/***************************************************************************
                          tnlGridCUDA2D.h  -  description
                             -------------------
    begin                : 2010/01/12
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

#ifndef tnlGridCUDA2DH
#define tnlGridCUDA2DH

#include <core/tnlFieldCUDA2D.h>
#include <diff/tnlGrid2D.h>

template< class T = double > class tnlGridCUDA2D :
	                         public tnlFieldCUDA2D< T >
{
   public:

   tnlGridCUDA2D( const char* name = 0 )
   {
      if( name )
         tnlFieldCUDA2D< T > :: SetName( name );
   };

   //! Constructor with the grid and the domain dimensions
   /*! @param x_size and @param y_size define the grid dimensions.
       @param A_x, @param B_x, @param A_y and @param B_y define domain
       Omega = <A_x,B_x>*<A_y,B_y>.
    */
   tnlGridCUDA2D( const char* name,
                  int x_size,
                  int y_size,
                  const double& A_x,
                  const double& B_x,
                  const double& A_y,
                  const double& B_y )
   : tnlFieldCUDA2D< T >( name, x_size, y_size ),
     Ax( A_x ), Bx( B_x ),
     Ay( A_y ), By( B_y )  
   {
      assert( Ax < Bx && Ay < By );
      Hx = ( Bx - Ax ) / ( double ) ( tnlFieldCUDA2D< T > :: GetXSize() - 1 );
      Hy = ( By - Ay ) / ( double ) ( tnlFieldCUDA2D< T > :: GetYSize() - 1 );
   };

   tnlGridCUDA2D( const tnlGridCUDA2D& g )
   : tnlFieldCUDA2D< T >( g ),
     Ax( g. Ax ), Bx( g. Bx ),
     Ay( g. Ay ), By( g. By ),
     Hx( g. Hx ), Hy( g. Hy )
   {
   };

   tnlGridCUDA2D( const tnlGrid2D< T >& g )
   : tnlFieldCUDA2D< T >( g ),
     Ax( g. GetAx() ), Bx( g. GetBx() ),
     Ay( g. GetAy() ), By( g. GetBy() ),
     Hx( g. GetHx() ), Hy( g. GetHy() )
   {
   };


   tnlString GetType() const
   {
      T t;
      return tnlString( "tnlGridCUDA2D< " ) + tnlString( GetParameterType( t ) ) + tnlString( " >" );
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
      if( ! hx ) Hx = ( Bx - Ax ) / ( double ) ( tnlFieldCUDA2D< T > :: GetXSize() - 1 );
      else Hx = hx;
      if( ! hy ) Hy = ( By - Ay ) / ( double ) ( tnlFieldCUDA2D< T > :: GetYSize() - 1 );
      else Hy = hy;
   }
   
   void SetNewDomain( const tnlGridCUDA2D< T >& u )
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
   
   protected:

   //! Domain dimensions
   double Ax, Bx, Ay, By;

   //! The grid space steps
   double Hx, Hy;
};

// Explicit instatiation
template class tnlGridCUDA2D< double >;

#endif
