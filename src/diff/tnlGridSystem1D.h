/***************************************************************************
                          tnlGridSystem1D.h  -  description
                             -------------------
    begin                : 2007/12/17
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

#ifndef tnlGridSystem1DH
#define tnlGridSystem1DH

#include <core/tnlFieldSystem1D.h>

template< typename T, int SYSTEM_SIZE, typename SYSTEM_INDEX > class tnlGridSystem1D : public tnlFieldSystem1D< T, SYSTEM_SIZE, SYSTEM_INDEX >
{
   public:

   tnlGridSystem1D() {};

   //! Constructor with the grid and the domain dimensions
   /*! @param x_size and @param y_size define the grid dimensions.
       @param A_x, @param B_x, @param A_y and @param B_y define domain
       Omega = <A_x,B_x>*<A_y,B_y>.
    */
   tnlGridSystem1D( long int x_size,
                  const double& A_x,
                  const double& B_x )
   : tnlFieldSystem1D< T, SYSTEM_SIZE, SYSTEM_INDEX >( x_size ),
     Ax( A_x ), Bx( B_x ) 
   {
      assert( Ax < Bx );
      Hx = ( Bx - Ax ) / ( double ) ( x_size - 1 ); 
   };

   tnlGridSystem1D( const tnlGridSystem1D& g )
   : tnlFieldSystem1D< T, SYSTEM_SIZE, SYSTEM_INDEX >( g ),
     Ax( g. Ax ), Bx( g. Bx ),
     Hx( g. Hx )
   { };

   tnlString GetType() const
   {
      T t;
      stringstream str;
      str << "tnlGridSystem1D< " << GetParameterType( t ) << ", " << SYSTEM_SIZE << " >";
      return tnlString( str. str(). data() );
   };

   void SetNewDomain( const double& A_x,
                      const double& B_x )
   {
      Ax = A_x;
      Bx = B_x;
      assert( Ax < Bx );
      Hx = ( Bx - Ax ) / ( double ) ( tnlFieldSystem1D< T, SYSTEM_SIZE, SYSTEM_INDEX > :: GetXSize() - 1 ); 
   }

   void SetNewDomain( const tnlGridSystem1D< T, SYSTEM_SIZE, SYSTEM_INDEX >& u )
   {
      SetNewDomain( u. GetAx(), u. GetBx() );
   }

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

   // Interpolation
   template< typename N > T Value( N x ) const
   {
      x = ( x - Ax ) / Hx;
      long int ix = ( long int ) ( x );
      N dx = x - ( N ) ix;
      assert( 0 );
   }
   
   //! Forward difference w.r.t x
   T Partial_x_f( const SYSTEM_INDEX ind, const long int i ) const
   {
      assert( i >= 0 );
      assert( i < ( tnlFieldSystem1D< T, SYSTEM_SIZE, SYSTEM_INDEX > :: x_size - 1 ) );
      return ( tnlFieldSystem1D< T, SYSTEM_SIZE, SYSTEM_INDEX > :: operator()( ind, i + 1 ) - 
               tnlFieldSystem1D< T, SYSTEM_SIZE, SYSTEM_INDEX > ::  operator()( ind, i ) ) / Hx;
   };
   
   //! Backward difference w.r.t x
   T Partial_x_b( const SYSTEM_INDEX ind, const long int i ) const
   {
      assert( i > 0 );
      assert( i < ( tnlFieldSystem1D< T, SYSTEM_SIZE, SYSTEM_INDEX > :: x_size ) );
      return ( tnlFieldSystem1D< T, SYSTEM_SIZE, SYSTEM_INDEX > :: operator()( ind, i ) - 
               tnlFieldSystem1D< T, SYSTEM_SIZE, SYSTEM_INDEX > :: operator()( ind, i - 1 ) ) / Hx;
   };

   //! Central difference w.r.t. x
   T Partial_x( const SYSTEM_INDEX ind, const long int i ) const
   {
      assert( i > 0 );
      assert( i < ( tnlFieldSystem1D< T, SYSTEM_SIZE, SYSTEM_INDEX > :: x_size - 1 ) );
      return ( tnlFieldSystem1D< T, SYSTEM_SIZE, SYSTEM_INDEX > :: operator()( ind, i + 1 ) - 
               tnlFieldSystem1D< T, SYSTEM_SIZE, SYSTEM_INDEX > :: operator()( ind, i - 1 ) ) / ( 2.0 * Hx );
   };
   
   //! Second order difference w.r.t. x
   T Partial_xx( const SYSTEM_INDEX ind, const long int i ) const
   {
      assert( i > 0 );
      assert( i < ( tnlFieldSystem1D< T, SYSTEM_SIZE, SYSTEM_INDEX > :: x_size - 1 ) );
      return ( tnlFieldSystem1D< T, SYSTEM_SIZE, SYSTEM_INDEX > :: operator()( ind, i + 1 ) - 
               2.0 * tnlFieldSystem1D< T, SYSTEM_SIZE, SYSTEM_INDEX > :: operator()( ind, i ) + 
               tnlFieldSystem1D< T, SYSTEM_SIZE, SYSTEM_INDEX > :: operator()( ind, i - 1 ) ) / ( Hx * Hx );   
   };
   
   //! Method for saving the object to a file as a binary data
   bool Save( ostream& file ) const
   {
      if( ! tnlFieldSystem1D< T, SYSTEM_SIZE, SYSTEM_INDEX > :: Save( file ) ) return false;
      file. write( ( char* ) &Ax, sizeof( double ) );
      file. write( ( char* ) &Bx, sizeof( double ) );
      file. write( ( char* ) &Hx, sizeof( double ) );
      if( file. bad() ) return false;
      return true;
   };

   //! Method for restoring the object from a file
   bool Load( istream& file )
   {
      if( ! tnlFieldSystem1D< T, SYSTEM_SIZE, SYSTEM_INDEX > :: Load( file ) ) return false;
      file. read( ( char* ) &Ax, sizeof( double ) );
      file. read( ( char* ) &Bx, sizeof( double ) );
      file. read( ( char* ) &Hx, sizeof( double ) );
      if( file. bad() ) return false;
      return true;
   };   

   protected:

   //! Domain dimensions
   double Ax, Bx;

   //! The grid space steps
   double Hx;
};

// Explicit instatiation
//template class tnlGridSystem1D< double, 1, int >;

#endif
