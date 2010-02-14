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

template< typename REAL >
bool compareGrid2D( const tnlString& first_file_uncompressed,
                    const tnlString& second_file_uncompressed,
                    bool write_difference,
                    int edge_skip,
                    const REAL& space_step,
                    REAL& l1_norm,
                    REAL& l2_norm,
                    REAL& max_norm,
                    REAL& h )
{
   dbgFunctionName( "", "compareGrid2D" );
   dbgCout( "Processing file with tnlGrid2D< REAL > ..." );
   dbgExpr( first_file_uncompressed );
   dbgExpr( second_file_uncompressed );
   tnlGrid2D< REAL > u1, u2, difference;
   fstream file;
   file. open( first_file_uncompressed. Data(), ios :: in | ios :: binary );
   if( ! u1. Load( file ) )
   {
      cerr << " unable to restore the data " << endl;
      file. close();
      return false;
   }
   file. close();
   file. open( second_file_uncompressed. Data(), ios :: in | ios :: binary );
   if( ! u2. Load( file ) )
   {
      cerr << " unable to restore the data " << endl;
      file. close();
      return false;
   }
   file. close();
   if( write_difference )
   {
      difference. SetNewDimensions( u1 );
      difference. SetNewDomain( u1 );
   }
   if( ! Compare( u1,
            u2,
            l1_norm,
            l2_norm,
            max_norm,
            difference,
            edge_skip) )
   {
      return false;
   }
   if( space_step ) h = space_step;
   else h = Min( u1. GetHx(), u1. GetHy() );
   return true;
}

#endif /* TNLERRNORMS_H_ */
