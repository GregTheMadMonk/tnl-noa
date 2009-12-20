/***************************************************************************
                          compare-objects.cpp  -  description
                             -------------------
    begin                : 2009/08/14
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

#include "compare-objects.h"

#include <math.h>
#include <core/mfuncs.h>

//--------------------------------------------------------------------------
bool Compare( const tnlGrid2D< double >& u1,
              const tnlGrid2D< double >& u2,
              double& l1_norm,
              double& l2_norm,
              double& max_norm,
              tnlGrid2D< double >& difference,
              int edge_skip )
{
   if( u1. GetXSize() != u2. GetXSize() ||
       u1. GetYSize() != u2. GetYSize() ||
       u1. GetAx()    != u2. GetAx()    ||
       u1. GetAy()    != u2. GetAy()    ||
       u1. GetBx()    != u2. GetBx()    ||
       u1. GetBy()    != u2. GetBy()    )
   {
      cerr << "Both grids describes different numerical domain." << endl;
      cerr << u1. GetXSize() << "x"  << u1. GetYSize( ) << " on domain < "
           << u1. GetAx()    << ", " << u1. GetBx() << " > x < "
           << u1. GetAy()    << ", " << u1. GetBy() << " > VS. "
           << u2. GetXSize() << "x"  << u2. GetYSize( ) << " on domain "
           << u2. GetAx()    << ", " << u2. GetBx() << " > x < "
           << u2. GetAy()    << ", " << u2. GetBy() << " > ." << endl;
      return false;
   }
   const int x_size = u1. GetXSize();
   const int y_size = u1. GetYSize();
   const double& hx = u1. GetHx();
   const double& hy = u1. GetHy();
   
   int i, j;
   l1_norm = l2_norm = max_norm = 0.0;
   for( i = edge_skip; i < x_size - edge_skip; i ++ )
      for( j = edge_skip; j < y_size - edge_skip; j ++ )
      {
         double diff = u1( i, j ) - u2( i, j );
         double err = fabs( diff );
         l1_norm += hx * hy * err;
         l2_norm += hx * hy * err * err;
         max_norm = Max( max_norm, err ); 
         if( difference ) difference( i, j ) = diff;
      }
   l2_norm = sqrt( l2_norm );
   return true; 
}
//--------------------------------------------------------------------------
bool Compare( const tnlGrid3D< double >& u1,
              const tnlGrid3D< double >& u2,
              double& l1_norm,
              double& l2_norm,
              double& max_norm,
              tnlGrid3D< double >& difference,
              int edge_skip )
{
   if( u1. GetXSize() != u2. GetXSize() ||
       u1. GetYSize() != u2. GetYSize() ||
       u1. GetZSize() != u2. GetZSize() ||
       u1. GetAx()    != u2. GetAx()    ||
       u1. GetAy()    != u2. GetAy()    ||
       u1. GetAz()    != u2. GetAz()    ||
       u1. GetBx()    != u2. GetBx()    ||
       u1. GetBy()    != u2. GetBy()    ||
       u1. GetBz()    != u2. GetBz()    )
   {
      cerr << "Both grids describes different numerical domain." << endl;
      return false;
   }
   const int x_size = u1. GetXSize();
   const int y_size = u1. GetYSize();
   const int z_size = u1. GetZSize();
   const double& hx = u1. GetHx();
   const double& hy = u1. GetHy();
   const double& hz = u1. GetHz();
   
   int i, j, k;
   l1_norm = l2_norm = max_norm = 0.0;
   for( i = edge_skip; i < x_size - edge_skip; i ++ )
      for( j = edge_skip; j < y_size - edge_skip; j ++ )
         for( k = edge_skip; k < z_size - edge_skip; k ++ )
         {
            double diff = u1( i, j, k ) - u2( i, j, k );
            double err = fabs( diff );
            l1_norm += hx * hy * hz * err;
            l2_norm += hx * hy * hz * err * err;
            max_norm = Max( max_norm, err ); 
            if( difference ) difference( i, j, k ) = diff;
         }
   l2_norm = sqrt( l2_norm );
   return true; 
}

