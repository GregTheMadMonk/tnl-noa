/***************************************************************************
                          tnlPGMImage_impl.h  -  description
                             -------------------
    begin                : Jul 20, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
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

#ifndef TNLPGMIMAGE_IMPL_H
#define	TNLPGMIMAGE_IMPL_H

#include <cstring>
#include <core/io/tnlPGMImage.h>

template< typename Index >
tnlPGMImage< Index >::
tnlPGMImage() : 
   binary( false ), colors( 0 )
{
}


template< typename Index >
bool
tnlPGMImage< Index >::
open( const tnlString& fileName )
{
   FILE* file = fopen( fileName.getString(), "r" );
   if( ! file )
   {
      cerr << "Unable to open the file " << fileName << endl;
      return false;
   }

   char magicNumber[ 3 ];
   magicNumber[ 2 ] = 0;
   if( fread( magicNumber, sizeof( char ), 2, file ) != 2 )
   {
      cerr << "Unable to read the magic number." << endl;
      return false;
   }

   cout << magicNumber << endl;
   if( strcmp( magicNumber, "P5" ) != 0 &&
       strcmp( magicNumber, "P2" ) != 0 )
      return false;
   
   if( strcmp( magicNumber, "P5" ) == 0 )
      this->binary = true;

   char line[ 1024 ];
   while( fread( line, sizeof( char ), 1, file ) &&
          ( line[ 0 ] == ' ' || line[ 0 ] == '\t' ||
            line[ 0 ] == '\r' || line[ 0 ] == '\n' ) );
   if( line[ 0 ] == '#' )
      while( fread( line, sizeof( char ), 1, file ) &&
             line[ 0 ] != '\n' );
   else fseek( file, -1, SEEK_CUR );

   fscanf( file, "%d %d\n", &this->width, &this->height );
   fscanf( file, "%d\n", &this->colors );
   return true;
}


#endif	/* TNLPGMIMAGE_IMPL_H */

