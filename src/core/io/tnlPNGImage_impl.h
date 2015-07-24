/***************************************************************************
                          tnlPNGImage_impl.h  -  description
                             -------------------
    begin                : Jul 24, 2015
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

#ifndef TNLPNGIMAGE_IMPL_H
#define	TNLPNGIMAGE_IMPL_H

#ifdef HAVE_PNG_H
#include <png.h>
#endif

template< typename Index >
tnlPNGImage< Index >::
tnlPNGImage() : 
   fileOpen( false )
{
}

template< typename Index >
bool
tnlPNGImage< Index >::
readHeader()
{
#ifdef HAVE_PNG_H
   /***
    * Check if it is a PNG image.
    */
   const int headerSize( 8 );
   png_byte header[ headerSize ];
   if( fread( header, sizeof( char ), headerSize, this->file ) != headerSize )
   {
      cerr << "I am not able to read PNG image header." << endl;
      return false;
   }
   bool isPNG = !png_sig_cmp( header, 0, headerSize );
   if( ! isPNG )
      return false;
   
   /****
    * Allocate necessary memory
    */
   this->png_ptr = png_create_read_struct( PNG_LIBPNG_VER_STRING,
                                           NULL,
                                           NULL,
                                           NULL );
   if( !this->png_ptr )
      return false;

   this->info_ptr = png_create_info_struct( this->png_ptr );
   if( !this->info_ptr )
   {
      png_destroy_read_struct( &this->png_ptr,
                              ( png_infopp ) NULL,
                              ( png_infopp ) NULL );
      return false;
   }

   this->end_info = png_create_info_struct( this->png_ptr );
   if( !this->end_info )
   {
      png_destroy_read_struct( &this->png_ptr,
                               &this->info_ptr,
                               ( png_infopp ) NULL );
      return false;
   }

   /***
    * Prepare the long jump back from libpng.
    */
   if( setjmp(png_jmpbuf( this->png_ptr ) ) )
   {
      png_destroy_read_struct( &this->png_ptr,
                               &this->info_ptr,
                               &end_info );
      return false;
   }
   png_init_io( this->png_ptr, this->file );
   png_set_sig_bytes( this->png_ptr, headerSize );
   
   /****
    * Read the header
    */
   png_read_png( this->png_ptr, this->info_ptr, PNG_TRANSFORM_IDENTITY, NULL );
   this->height = this->info_ptr->height; //( Index ) png_get_image_height( this->png_ptr, this->info_ptr );
   this->width = this->info_ptr->width; //( Index ) png_get_image_width( this->png_ptr, this->info_ptr );
   this->bit_depth = png_get_bit_depth( this->png_ptr, this->info_ptr );
   this->color_type = png_get_color_type( this->png_ptr, this->info_ptr );
   cout << this->height << " x " << this->width << endl;
   return true;   
#else
   cerr << "TNL was not compiled with support of PNG. You may still use PGM format." << endl;
   return false;
#endif   
}

template< typename Index >
bool
tnlPNGImage< Index >::
openForRead( const tnlString& fileName )
{
   this->close();
   this->file = fopen( fileName.getString(), "r" );
   if( ! this->file )
   {
      cerr << "Unable to open the file " << fileName << endl;
      return false;
   }
   this->fileOpen = true;
   if( ! readHeader() )
      return false;
   return true;
}

template< typename Index >
   template< typename Real,
             typename Device,
             typename Vector >
bool
tnlPNGImage< Index >::
read( const tnlRegionOfInterest< Index > roi,
      const tnlGrid< 2, Real, Device, Index >& grid,
      Vector& vector )
{
#ifdef HAVE_PNG_H
   typedef tnlGrid< 2, Real, Device, Index > GridType;
   typedef typename GridType::CoordinatesType CoordinatesType;
   
   /***
    * Prepare the long jump back from libpng.
    */
   if( setjmp(png_jmpbuf( this->png_ptr ) ) )
   {
      png_destroy_read_struct( &this->png_ptr,
                               &this->info_ptr,
                               &this->end_info );
      return false;
   }
   
   png_bytepp row_pointers = png_get_rows( this->png_ptr, this->info_ptr );
   
   Index i, j;
   for( i = 0; i < this->height; i ++ )
   {
      for( j = 0; j < this->width; j ++ )
      {
         if( !roi.isIn( i, j ) )
            continue;
      
         Index cellIndex = grid.getCellIndex( CoordinatesType( j - roi.getLeft(),
                                              roi.getBottom() - 1 - i ) );
         unsigned char char_color[ 4 ];
         unsigned int int_color[ 4 ];
         switch( this->color_type )
         {
            case PNG_COLOR_TYPE_GRAY:
               if( this->bit_depth == 8 )
               {
                  char_color[ 0 ] = row_pointers[ i ][ j ];
                  Real value = char_color[ 0 ] / ( Real ) 255.0;
                  vector.setElement( cellIndex, value );
               }
               if( this->bit_depth == 16 )
               {
                  int_color[ 0 ] = row_pointers[ i ][ j ];
                  Real value = int_color[ 0 ] / ( Real ) 65535.0;
                  vector.setElement( cellIndex, value );
               }
               break;
            case PNG_COLOR_TYPE_RGB:
               if( this->bit_depth == 8 )
               {
                  char_color[ 0 ] = row_pointers[ i ][ 3 * j ];
                  char_color[ 1 ] = row_pointers[ i ][ 3 * j + 1 ];
                  char_color[ 2 ] = row_pointers[ i ][ 3 * j + 2 ];
                  Real r = char_color[ 0 ] / ( Real ) 255.0;
                  Real g = char_color[ 1 ] / ( Real ) 255.0;
                  Real b = char_color[ 2 ] / ( Real ) 255.0;
                  Real value = 0.2989 * r + 0.5870 * g + 0.1140 * b;
                  vector.setElement( cellIndex, value );
               }
               if( this->bit_depth == 16 )
               {
                  int_color[ 0 ] = row_pointers[ i ][ 3 * j ];
                  int_color[ 1 ] = row_pointers[ i ][ 3 * j + 1 ];
                  int_color[ 2 ] = row_pointers[ i ][ 3 * j + 2 ];
                  Real r = int_color[ 0 ] / ( Real ) 65535.0;
                  Real g = int_color[ 1 ] / ( Real ) 66355.0;
                  Real b = int_color[ 2 ] / ( Real ) 65535.0;
                  Real value = 0.2989 * r + 0.5870 * g + 0.1140 * b;
                  vector.setElement( cellIndex, value );
               }
               break;
            default:
               cerr << "Unknown PNG color type." << endl;
               return false;
         }
      }
   }
   return true;
#else
   cerr << "TNL was not compiled with support of PNG. You may still use PGM format." << endl;
   return false;
#endif      
}

template< typename Index >
   template< typename Real,
             typename Device >
bool
tnlPNGImage< Index >::
writeHeader( const tnlGrid< 2, Real, Device, Index >& grid,
             bool binary )
{
#idef HAVE_PNG_H
   this->png_ptr = png_create_write_struct( PNG_LIBPNG_VER_STRING,
                                            NULL,
                                            NULL,
                                            NULL );
   if( !png_ptr )
      return false;

   this->info_ptr = png_create_info_struct( this->png_ptr );
   if( !this->info_ptr )
   {
      png_destroy_write_struct( &this->png_ptr,
                                NULL);
      return false;
   }
   
#else
   cerr << "TNL was not compiled with support of PNG. You may still use PGM format." << endl;
   return false;
#endif   
   
}

template< typename Index >
   template< typename Real,
             typename Device >
bool
tnlPNGImage< Index >::
openForWrite( const tnlString& fileName,
              tnlGrid< 2, Real, Device, Index >& grid )
{
   this->close();
   this->file = fopen( fileName.getString(), "w" );
   if( ! this->file )
   {
      cerr << "Unable to open the file " << fileName << endl;
      return false;
   }
   this->fileOpen = true;
   if( ! writeHeader( grid ) )
      return false;
   return true;
}


template< typename Index >
void
tnlPNGImage< Index >::
close()
{
   if( this->fileOpen )
      fclose( file );
   this->fileOpen = false;
}

template< typename Index >
tnlPNGImage< Index >::
~tnlPNGImage()
{
   close();
}


#endif	/* TNLPNGIMAGE_IMPL_H */

