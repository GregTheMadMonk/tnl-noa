/***************************************************************************
                          tnlJPEGImage_impl.h  -  description
                             -------------------
    begin                : Jul 25, 2015
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

#ifndef TNLJPEGIMAGE_IMPL_H
#define	TNLJPEGIMAGE_IMPL_H

#include <core/io/tnlJPEGImage.h>
#include <setjmp.h>

inline void my_error_exit( j_common_ptr cinfo )
{
  /* cinfo->err really points to a my_error_mgr struct, so coerce pointer */
  my_error_mgr* myerr = ( my_error_mgr* ) cinfo->err;

  /* Always display the message. */
  /* We could postpone this until after returning, if we chose. */
  ( *cinfo->err->output_message )( cinfo );

  /* Return control to the setjmp point */
  longjmp( myerr->setjmp_buffer, 1 );
}

template< typename Index >
tnlJPEGImage< Index >::
tnlJPEGImage() : 
   fileOpen( false )
{
}

template< typename Index >
bool
tnlJPEGImage< Index >::
readHeader()
{
#ifdef HAVE_JPEG_H
   this->cinfo.err = jpeg_std_error(&jerr.pub);
   this->jerr.pub.error_exit = my_error_exit; 
   
   /***
    * Prepare the long jump back from libjpeg.
    */
   if( setjmp( jerr.setjmp_buffer ) )
   {
       /****
        * If we get here, the JPEG code has signaled an error.
        * We need to clean up the JPEG object, close the input file, and return.
        */
      jpeg_destroy_decompress( &this->cinfo );
      return false;
   }
   
   jpeg_create_decompress( &this->cinfo );
   jpeg_stdio_src( &this->cinfo, this->file );
   if( jpeg_read_header( &this->cinfo, true ) != JPEG_HEADER_OK )
      return false;
   this->height = this->cinfo.image_height;
   this->width = this->cinfo.image_width;
   this->components = this->cinfo.num_components; 
   //this->color_space = this->cinfo.jpeg_color_space;
   //cout << this->height << " x " << this->width << " : " << this->components << " " << this->color_space << endl;
#else
   cerr << "TNL was not compiled with support of JPEG. You may still use PGM format." << endl;
   return false;
#endif   
}

template< typename Index >
bool
tnlJPEGImage< Index >::
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
tnlJPEGImage< Index >::
read( const tnlRegionOfInterest< Index > roi,
      const tnlGrid< 2, Real, Device, Index >& grid,
      Vector& vector )
{
#ifdef HAVE_PNG_H
   typedef tnlGrid< 2, Real, Device, Index > GridType;
   typedef typename GridType::CoordinatesType CoordinatesType;
   
   /***
    * Prepare the long jump back from libjpeg.
    */
   if( setjmp( jerr.setjmp_buffer ) )
   {
       /****
        * If we get here, the JPEG code has signaled an error.
        * We need to clean up the JPEG object, close the input file, and return.
        */
      jpeg_destroy_decompress( &this->cinfo );
      return false;
   }
      
   jpeg_start_decompress( &this->cinfo );
   int row_stride = this->cinfo.output_width * this->cinfo.output_components;
   JSAMPARRAY row = ( *( this->cinfo.mem->alloc_sarray ) )( ( j_common_ptr ) &this->cinfo,
                                                            JPOOL_IMAGE,
                                                            row_stride,
                                                            1 );	
   
   Index i, j;
   while( this->cinfo.output_scanline < this->cinfo.output_height)
   {
      jpeg_read_scanlines( &this->cinfo, row, 1 );
      for( j = 0; j < this->width; j ++ )
      {
         if( !roi.isIn( i, j ) )
            continue;
      
         Index cellIndex = grid.getCellIndex( CoordinatesType( j - roi.getLeft(),
                                              roi.getBottom() - 1 - i ) );
         unsigned char char_color[ 4 ];
         unsigned int int_color[ 4 ];
         Real value, r, g, b;
         switch( this->components )
         {
            case 1:
               char_color[ 0 ] = row[ 0 ][ j ];
               value = char_color[ 0 ] / ( Real ) 255.0;
               vector.setElement( cellIndex, value );
               break;
            case 3:
               char_color[ 0 ] = row[ 0 ][ 3 * j ];
               char_color[ 1 ] = row[ 0 ][ 3 * j + 1 ];
               char_color[ 2 ] = row[ 0 ][ 3 * j + 2 ];
               r = char_color[ 0 ] / ( Real ) 255.0;
               g = char_color[ 1 ] / ( Real ) 255.0;
               b = char_color[ 2 ] / ( Real ) 255.0;
               value = 0.2989 * r + 0.5870 * g + 0.1140 * b;
               vector.setElement( cellIndex, value );
               break;
            default:
               cerr << "Unknown JPEG color type." << endl;
               return false;
         }
      }
   }
   return true;
#else
   //cerr << "TNL was not compiled with support of JPEG. You may still use PGM format." << endl;
   return false;
#endif      
}

template< typename Index >
   template< typename Real,
             typename Device >
bool
tnlJPEGImage< Index >::
writeHeader( const tnlGrid< 2, Real, Device, Index >& grid )
{
#ifdef HAVE_JPEG_H
#else
   cerr << "TNL was not compiled with support of PNG. You may still use PGM format." << endl;
   return false;
#endif    
}

template< typename Index >
   template< typename Real,
             typename Device >
bool
tnlJPEGImage< Index >::
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
   template< typename Real,
             typename Device,
             typename Vector >
bool
tnlJPEGImage< Index >::
write( const tnlGrid< 2, Real, Device, Index >& grid,
       Vector& vector )
{
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
              
   Index i, j;
   png_bytep row = new png_byte[ 3 * grid.getDimensions().x() ];
   for( i = 0; i < grid.getDimensions().y(); i ++ )
   {
      for( j = 0; j < grid.getDimensions().x(); j ++ )
      {
         Index cellIndex = grid.getCellIndex( CoordinatesType( j,
                                              grid.getDimensions().y() - 1 - i ) );

         row[ j ] = 255 * vector.getElement( cellIndex );         
      }
      png_write_row( this->png_ptr, row );
   }
   //png_set_rows( this->png_ptr, this->info_ptr, row_pointers );
   //png_write_png( this->png_ptr, this->info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
   delete[] row;
   return true;
}


template< typename Index >
void
tnlJPEGImage< Index >::
close()
{
   if( this->fileOpen )
      fclose( file );
   this->fileOpen = false;
}

template< typename Index >
tnlJPEGImage< Index >::
~tnlJPEGImage()
{
   close();
}



#endif	/* TNLJPEGIMAGE_IMPL_H */

