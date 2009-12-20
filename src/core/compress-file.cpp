/***************************************************************************
                          compress-file.cpp  -  description
                             -------------------
    begin                : 2007/07/02
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

#include <iostream>
#include <stdlib.h>
#include <cstring>
#include <tnlString.h>
#include "compress-file.h"


//--------------------------------------------------------------------------
bool CompressFile( const char* file_name, const char* format )
{
   //cout << "Compressing file " << file_name << endl;
   tnlString command;
   if( strcmp( format, "gz" ) == 0 )
      command. SetString( "gzip " );
   if( strcmp( format, "bz2" ) == 0 )
      command. SetString( "bzip2 " );
   command += "--best --force ";
   command += file_name;
   if( system( command. Data() ) != 0 )
   {
      cerr << "Some error appeared when when processing the command: " << command << endl;
      return false;
   }
   return true;
}
//--------------------------------------------------------------------------
bool UnCompressFile( const char* file_name, const char* format )
{
   tnlString command;
   if( strcmp( format, "gz" ) == 0 )
      command. SetString( "gunzip " );
   if( strcmp( format, "bz2" ) == 0 )
      command. SetString( "bunzip2 " );
   command += file_name;
   if( system( command. Data() ) == -1 )
   {
      cerr << "Some error appeared when when processing the command: " << command << endl;
      return false;
   }
   return true;
}

