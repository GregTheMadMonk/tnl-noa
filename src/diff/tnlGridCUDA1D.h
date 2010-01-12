/***************************************************************************
                          tnlGrid1D.h  -  description
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

#ifndef tnlGridCUDA1DH
#define tnlGridCUDA1DH

#include <core/tnlFieldCUDA1D.h>

template< typename T = double > class tnlGridCUDA1D : public tnlFieldCUDA1D< T >
{
   public:

   tnlGridCUDA1D()
   {
   };

   //! Constructor with the grid and the domain dimensions
   /*! @param x_size and @param y_size define the grid dimensions.
       @param A_x, @param B_x, @param A_y and @param B_y define domain
       Omega = <A_x,B_x>*<A_y,B_y>.
    */
   tnlGridCUDA1D( int x_size,
                  const double& A_x,
                  const double& B_x )
   : tnlFieldCUDA1D< T >( x_size ),
     Ax( A_x ), Bx( B_x ) 
   {
      assert( Ax < Bx );
      Hx = ( Bx - Ax ) / ( double ) ( x_size - 1 ); 
   };

   tnlGridCUDA1D( const tnlGridCUDA1D& g )
   : tnlFieldCUDA1D< T >( g ),
     Ax( g. Ax ), Bx( g. Bx ),
     Hx( g. Hx )
   { };

   tnlString GetType() const
   {
      T t;
      return tnlString( "tnlGridCUDA1D< " ) + tnlString( GetParameterType( t ) ) + tnlString( " >" );
   };

   void SetNewDomain( const double& A_x,
                      const double& B_x,
                      double hx = 0.0 )
   {
      Ax = A_x;
      Bx = B_x;
      assert( Ax < Bx );
      if( ! hx ) Hx = ( Bx - Ax ) / ( double ) ( tnlFieldCUDA1D< T > :: GetXSize() - 1 );
      else Hx = hx;
   }
   ;
   void SetNewDomain( const tnlGridCUDA1D< T >& u )
   {
      SetNewDomain( u. GetAx(),
                    u. GetBx() );
   };

   const double& GetAx() const
   {
      return Ax;
   }

   const double& GetBx() const
   {
      return Bx;
   }

   const double& GetHx() const
   {
      return Hx;
   }

   protected:

   //! Domain dimensions
   double Ax, Bx;

   //! The grid space steps
   double Hx;
};

// Explicit instatiation
template class tnlGridCUDA1D< double >;

#endif
