/***************************************************************************
                          compare-objects.h  -  description
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

#ifndef compare_objectsH
#define compare_objectsH

#include <legacy/mesh/tnlGridOld.h>
#include <math.h>
#include <core/mfuncs.h>

template< typename Real1,
          typename Real2,
          typename Device1,
          typename Device2,
          typename Index >
bool compareObjects( const tnlGridOld< 2, Real1, Device1, Index >& u1,
                     const tnlGridOld< 2, Real2, Device2, Index >& u2,
                     Real1& l1_norm,
                     Real1& l2_norm,
                     Real1& max_norm,
                     tnlGridOld< 2, Real1, Device1, Index >& difference,
                     Index edge_skip )
{
   if( u1. getDimensions() != u2. getDimensions() ||
       u1. getDomainLowerCorner() != u2. getDomainLowerCorner() ||
       u1. getDomainUpperCorner() != u2. getDomainUpperCorner() )
   {
      cerr << "Both grids describes different numerical domain." << endl;
      //TODO: Fix this - it produces compile errors.
      /*cerr << u1. getDimensions() << " on domain "
           << u1. getDomainLowerCorner() << " -- "
           << u1. getDomainUpperCorner()  
           << u2. getDimensions() << " on domain "
           << u2. getDomainLowerCorner << " -- "
           << u2. getDomainUpperCorner() << endl;*/
      return false;
   }
   const Index xSize = u1. getDimensions(). x();
   const Index ySize = u1. getDimensions(). y();
   const Real1& hx = u1. getSpaceSteps(). x();
   const Real1& hy = u1. getSpaceSteps(). y();

   l1_norm = l2_norm = max_norm = 0.0;
   for( Index j = edge_skip; j < ySize - edge_skip; j ++ )
      for( Index i = edge_skip; i < xSize - edge_skip; i ++ )
      {
         Real1 diff = u1. getElement( j, i ) - Real1( u2. getElement( j, i ) );
         Real1 err = fabs( diff );
         l1_norm += hx * hy * err;
         l2_norm += hx * hy * err * err;
         max_norm = Max( max_norm, err );
         if( difference ) difference. setElement( j, i, diff );
      }
   l2_norm = sqrt( l2_norm );
   return true;
}

template< typename Real1,
          typename Real2,
          typename Device1,
          typename Device2,
          typename Index >
bool compareObjects( const tnlGridOld< 3, Real1, Device1, Index >& u1,
                     const tnlGridOld< 3, Real2, Device2, Index >& u2,
                     Real1& l1_norm,
                     Real1& l2_norm,
                     Real1& max_norm,
                     tnlGridOld< 3, Real1, Device1, Index >& difference,
                     Index edge_skip )
{
   if( u1. getDimensions() != u2. getDimensions() ||
       u1. getDomainLowerCorner() != u2. getDomainLowerCorner() ||
       u1. getDomainUpperCorner() != u2. getDomainUpperCorner() )
   {
      cerr << "Both grids describes different numerical domain." << endl;
      //TODO: Fix this - it produces compile errors.
      /*cerr << u1. getDimensions() << " on domain "
           << u1. getDomainLowerCorner() << " -- "
           << u1. getDomainUpperCorner()  
           << u2. getDimensions() << " on domain "
           << u2. getDomainLowerCorner << " -- "
           << u2. getDomainUpperCorner() << endl;*/
      return false;
   }
   const Index xSize = u1. getDimensions(). x();
   const Index ySize = u1. getDimensions(). y();
   const Index zSize = u1. getDimensions(). z();
   const Real1& hx = u1. getSpaceSteps(). x();
   const Real1& hy = u1. getSpaceSteps(). y();
   const Real1& hz = u1. getSpaceSteps(). z();

   l1_norm = l2_norm = max_norm = 0.0;
   for( Index k = edge_skip; k < zSize - edge_skip; k ++ )
      for( Index j = edge_skip; j < ySize - edge_skip; j ++ )
         for( Index i = edge_skip; i < xSize - edge_skip; i ++ )
         {
            Real1 diff = u1. getElement( k, j, i ) - Real1( u2. getElement( k, j, i ) );
            Real1 err = fabs( diff );
            l1_norm += hx * hy * err;
            l2_norm += hx * hy * err * err;
            max_norm = Max( max_norm, err );
            if( difference ) difference. setElement( k, j, i, diff );
         }
   l2_norm = sqrt( l2_norm );
   return true;
}

#endif
