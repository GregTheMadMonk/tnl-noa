/***************************************************************************
                          mfilename.cpp  -  description
                             -------------------
    begin                : 2007/06/18
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

#include <cstring>
#include <cstdlib>
#include "mfilename.h"
#include "tnlString.h"
#include "mfuncs.h" 

//--------------------------------------------------------------------------
void FileNameBaseNumberEnding( const char* base_name,
                               int number,
                               int index_size,
                               const char* ending,
                               tnlString& file_name )
{
   file_name. SetString( base_name );
   char snumber[ 1024 ], zeros[ 1024 ];;
   sprintf( snumber, "%d", number );
   int len = strlen( snumber );

   const int k = Min( 1024, index_size );
   int i;
   for( i = len; i < k ; i ++ ) 
      zeros[ i - len ] = '0';
   zeros[ k - len ] = 0;
   file_name += zeros;
   file_name += snumber;
   file_name += ending;
}
//--------------------------------------------------------------------------
void RemoveFileExtension( tnlString& file_name )
{
   int size = file_name. Length();
   int i = 1;
   while( file_name. Data()[ size - i ] != '.' && size > i  ) i ++ ;
   file_name. SetString( file_name. Data(), 0, i );
}
