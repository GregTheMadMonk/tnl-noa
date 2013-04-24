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
#include <core/mfilename.h>
#include <core/tnlString.h>
#include <core/mfuncs.h>

//--------------------------------------------------------------------------
void FileNameBaseNumberEnding( const char* base_name,
                               int number,
                               int index_size,
                               const char* ending,
                               tnlString& file_name )
{
   file_name. setString( base_name );
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
   int size = file_name. getLength();
   int i = 1;
   while( file_name. getString()[ size - i ] != '.' && size > i  ) i ++ ;
   file_name. setString( file_name. getString(), 0, i );
}
