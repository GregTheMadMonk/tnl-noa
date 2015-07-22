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
   binary( false ), maxColors( 0 ), fileOpen( false )
{
}

template< typename Index >
bool
tnlPGMImage< Index >::
readHeader( FILE* file )
{
   char magicNumber[ 3 ];
   magicNumber[ 2 ] = 0;
   if( fread( magicNumber, sizeof( char ), 2, file ) != 2 )
   {
      cerr << "Unable to read the magic number." << endl;
      return false;
   }

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
   fscanf( file, "%d\n", &this->maxColors );
   return true;   
}

template< typename Index >
bool
tnlPGMImage< Index >::
openForRead( const tnlString& fileName )
{
   this->file = fopen( fileName.getString(), "r" );
   if( ! this->file )
   {
      cerr << "Unable to open the file " << fileName << endl;
      return false;
   }
   this->fileOpen = true;
   if( ! readHeader( this->file ) )
      return false;
   return true;
}

template< typename Index >
   template< typename Real,
             typename Device,
             typename Vector >
bool
tnlPGMImage< Index >::
read( const tnlRegionOfInterest< Index > roi,
      const tnlGrid< 2, Real, Device, Index >& grid,
      Vector& vector )
{
   typedef tnlGrid< 2, Real, Device, Index > GridType;
   typedef typename GridType::CoordinatesType CoordinatesType;
   
   Index i, j;
   for( i = 0; i < this->height; i ++ )
      for( j = 0; j < this->width; j ++ )
      {
         int col;
         if( this->binary ) col = getc( this->file );
         else fscanf( this->file, "%d", &col );
         if( roi.isIn( i, j ) )
         {
            Index cellIndex = grid.getCellIndex( CoordinatesType( j - roi.getLeft(),
                                                                  roi.getBottom() - 1 - i ) );
            vector.setElement( cellIndex, ( Real ) col / ( Real ) this->maxColors );
         }
      }
   return true;
}

template< typename Index >
void
tnlPGMImage< Index >::
close()
{
   if( this->fileOpen )
      fclose( file );
   this->fileOpen = false;
}

template< typename Index >
tnlPGMImage< Index >::
~tnlPGMImage()
{
   close();
}

#endif	/* TNLPGMIMAGE_IMPL_H */

