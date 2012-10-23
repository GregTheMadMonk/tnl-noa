/***************************************************************************
                          tnl-err-norms.h  -  description
                             -------------------
    begin                : Feb 10, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
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

#ifndef TNLERRNORMS_H_
#define TNLERRNORMS_H_

#include <debug/tnlDebug.h>
#include <core/tnlString.h>
#include "compare-objects.h"

template< int Dimensions,
          typename Real1,
          typename Real2,
          typename Device1,
          typename Device2,
          typename Index >
bool compareGrids( const tnlString& firstFileName,
                   const tnlString& secondFileName,
                   bool write_difference,
                   int edge_skip,
                   const Real1& space_step,
                   Real1& l1_norm,
                   Real1& l2_norm,
                   Real1& max_norm,
                   Real1& h,
                   bool verbose )
{
   dbgFunctionName( "", "compareGrids" );
   dbgCout( "Processing file with tnlGrid ..." );
   dbgExpr( firstFileName );
   dbgExpr( secondFileName );
   tnlGrid< Dimensions, Real1, Device1, Index > u1( "u1" ), difference( "difference" );
   tnlGrid< Dimensions, Real2, Device2, Index > u2( "u2" );

   tnlFile file;
   file. open( firstFileName, tnlReadMode );
   if( ! u1. load( file ) )
   {
      cerr << " unable to restore the data " << endl;
      file. close();
      return false;
   }
   file. close();
   file. open( secondFileName, tnlReadMode );
   if( ! u2. load( file ) )
   {
      cerr << " unable to restore the data " << endl;
      file. close();
      return false;
   }
   file. close();
   if( write_difference )
   {
      if( ! difference. setLike( u1 ) )
      {
         cerr << "I do not have enough memory to allocate the differencing grid." << endl;
         return false;
      }
      difference. setValue( 0.0 );
   }
   if( ! compareObjects( u1,
                         u2,
                         l1_norm,
                         l2_norm,
                         max_norm,
                         difference,
                         edge_skip) )
      return false;
   if( write_difference )
   {
      tnlString fileName = firstFileName ;
      fileName += tnlString( ".diff.tnl" );
      if( verbose )
         cout << endl << "Writing the difference grid to " << fileName << endl;
      difference. save( fileName );
   }
   if( space_step != Real1( 0.0 ) ) h = space_step;
   else h = Min( u1. getSpaceSteps(). x(), u1. getSpaceSteps(). y() );
   return true;
}

#endif /* TNLERRNORMS_H_ */
