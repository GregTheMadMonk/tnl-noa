/***************************************************************************
                          tnlJPEGImage.h  -  description
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

#ifndef TNLJPEGIMAGE_H
#define	TNLJPEGIMAGE_H

#include <tnlConfig.h>

#ifdef HAVE_JPEG_H
#include <jpeglib.h>
#endif

#include <core/tnlString.h>
#include <core/io/tnlImage.h>
#include <core/io/tnlRegionOfInterest.h>

#ifdef HAVE_JPEG_H      
struct my_error_mgr
{
   jpeg_error_mgr pub;
   jmp_buf setjmp_buffer;
};
#endif

template< typename Index = int >
class tnlJPEGImage : public tnlImage< Index >
{
   public:
      
      typedef Index IndexType;
      
      tnlJPEGImage();
       
      bool openForRead( const tnlString& fileName );
      
      template< typename Real,
                typename Device,
                typename Vector >
      bool read( const tnlRegionOfInterest< Index > roi,
                 const tnlGrid< 2, Real, Device, Index >& grid,
                 Vector& vector );
      
      template< typename Real,
                typename Device >
      bool openForWrite( const tnlString& fileName,
                         tnlGrid< 2, Real, Device, Index >& grid );
      
      template< typename Real,
                typename Device,
                typename Vector >
      bool write( const tnlGrid< 2, Real, Device, Index >& grid,
                  Vector& vector );
      
      void close();
      
      ~tnlJPEGImage();
      
   protected:
      
      bool readHeader();
         
      template< typename Real,
                typename Device >
      bool writeHeader( const tnlGrid< 2, Real, Device, Index >& grid );
    
      FILE* file;

      bool fileOpen;

#ifdef HAVE_JPEG_H      
      my_error_mgr jerr;
      jpeg_decompress_struct cinfo;
      int components;
      J_COLOR_SPACE color_space;
#endif         
};

#include <core/io/tnlJPEGImage_impl.h>


#endif	/* TNLJPEGIMAGE_H */

