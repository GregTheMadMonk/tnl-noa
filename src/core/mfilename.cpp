/***************************************************************************
                          mfilename.cpp  -  description
                             -------------------
    begin                : 2007/06/18
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <cstring>
#include <cstdlib>
#include <core/mfilename.h>
#include <core/tnlString.h>
#include <core/mfuncs.h>

namespace TNL {

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

   const int k = min( 1024, index_size );
   int i;
   for( i = len; i < k ; i ++ )
      zeros[ i - len ] = '0';
   zeros[ k - len ] = 0;
   file_name += zeros;
   file_name += snumber;
   file_name += ending;
}

tnlString getFileExtension( const tnlString fileName )
{
   int size = fileName. getLength();
   int i = 1;
   while( fileName. getString()[ size - i ] != '.' && size > i  ) i ++ ;
   tnlString result;
   result.setString( fileName. getString(), size - i + 1 );
   return result;
}

void RemoveFileExtension( tnlString& fileName )
{
   int size = fileName. getLength();
   int i = 1;
   while( fileName. getString()[ size - i ] != '.' && size > i  ) i ++ ;
   fileName. setString( fileName. getString(), 0, i );
}

} // namespace TNL