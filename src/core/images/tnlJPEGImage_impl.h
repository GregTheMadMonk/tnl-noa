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

#include <core/images/tnlJPEGImage.h>
#include <setjmp.h>

#ifdef HAVE_JPEG_H
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
#endif

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
   this->decinfo.err = jpeg_std_error(&jerr.pub);
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
      jpeg_destroy_decompress( &this->decinfo );
      return false;
   }
   
   jpeg_create_decompress( &this->decinfo );
   jpeg_stdio_src( &this->decinfo, this->file );
   if( jpeg_read_header( &this->decinfo, true ) != JPEG_HEADER_OK )
      return false;
   this->height = this->decinfo.image_height;
   this->width = this->decinfo.image_width;
   this->components = this->decinfo.num_components; 
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
#ifdef HAVE_JPEG_H
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
      jpeg_destroy_decompress( &this->decinfo );
      return false;
   }
      
   jpeg_start_decompress( &this->decinfo );
   int row_stride = this->decinfo.output_width * this->decinfo.output_components;
   JSAMPARRAY row = ( *( this->decinfo.mem->alloc_sarray ) )( ( j_common_ptr ) &this->decinfo,
                                                              JPOOL_IMAGE,
                                                              row_stride,
                                                              1 );	
   
   Index i( 0 ), j;
   while( this->decinfo.output_scanline < this->decinfo.output_height)
   {
      jpeg_read_scanlines( &this->decinfo, row, 1 );
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
      i++;
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
   this->cinfo.err = jpeg_std_error( &this->jerr.pub );
   jpeg_create_compress( &this->cinfo );
   jpeg_stdio_dest( &this->cinfo, this->file );
   this->cinfo.image_width = grid.getDimensions().x();
   this->cinfo.image_height = grid.getDimensions().y();
   this->cinfo.input_components = 1;
   this->cinfo.in_color_space = JCS_GRAYSCALE;
   jpeg_set_defaults( &this->cinfo );
   jpeg_start_compress( &this->cinfo, true );
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

#ifdef HAVE_JPEG_H   
   Index i( 0 ), j;
   JSAMPROW row[1];
   row[ 0 ] = new JSAMPLE[ grid.getDimensions().x() ];
   // JSAMPLE is unsigned char
   while( this->cinfo.next_scanline < this->cinfo.image_height )
   {
      for( j = 0; j < grid.getDimensions().x(); j ++ )
      {
         Index cellIndex = grid.getCellIndex( CoordinatesType( j,
                                              grid.getDimensions().y() - 1 - i ) );

         row[ 0 ][ j ] = 255 * vector.getElement( cellIndex );         
      }
      jpeg_write_scanlines( &this->cinfo, row, 1 );
      i++;
   }   
   jpeg_finish_compress( &this->cinfo );
   jpeg_destroy_compress( &this->cinfo );
   delete[] row[ 0 ];
   return true;
#else
   return false;
#endif   
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

