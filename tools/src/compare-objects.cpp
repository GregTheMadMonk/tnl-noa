/***************************************************************************
                          compare-objects.cpp  -  description
                             -------------------
    begin                : 2009/08/14
    copyright            : (C) 2009 by Tomas Oberhuber
    email                : oberhuber@seznam.cz
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
bool Compare( const mGrid2D< double >& u1,
              const mGrid2D< double >& u2,
              double& l1_norm,
              double& l2_norm,
              double& max_norm,
              mGrid2D< double >& difference )
{
   if( u1. GetXSize() != u2. GetXSize() ||
       u1. GetYSize() != u2. GetYSize() ||
       u1. GetAx()    != u2. GetAx()    ||
       u1. GetAy()    != u2. GetAy()    ||
       u1. GetBx()    != u2. GetBx()    ||
       u1. GetBy()    != u2. GetBy()    )
   {
      cerr << "Both grids describes different numerical domain." << endl;
      return false;
   }
   const long int x_size = u1. GetXSize();
   const long int y_size = u1. GetYSize();
   const double& hx = u1. GetHx();
   const double& hy = u1. GetHy();
   
   long int i, j;
   l1_norm = l2_norm = max_norm = 0.0;
   for( i = 0; i < x_size; i ++ )
      for( j = 0; j < y_size; j ++ )
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
bool Compare( const mGrid3D< double >& u1,
              const mGrid3D< double >& u2,
              double& l1_norm,
              double& l2_norm,
              double& max_norm,
              mGrid3D< double >& difference )
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
   const long int x_size = u1. GetXSize();
   const long int y_size = u1. GetYSize();
   const long int z_size = u1. GetZSize();
   const double& hx = u1. GetHx();
   const double& hy = u1. GetHy();
   const double& hz = u1. GetHz();
   
   long int i, j, k;
   l1_norm = l2_norm = max_norm = 0.0;
   for( i = 0; i < x_size; i ++ )
      for( j = 0; j < y_size; j ++ )
         for( k = 0; k < z_size; k ++ )
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

