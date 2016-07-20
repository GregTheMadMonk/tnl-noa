/***************************************************************************
                          tnlJPEGImage.h  -  description
                             -------------------
    begin                : Jul 25, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/tnlConfig.h>

#ifdef HAVE_JPEG_H
#include <jpeglib.h>
#endif

#include <TNL/String.h>
#include <TNL/core/images/tnlImage.h>
#include <TNL/core/images/tnlRegionOfInterest.h>

#ifdef HAVE_JPEG_H
struct my_error_mgr
{
   jpeg_error_mgr pub;
   jmp_buf setjmp_buffer;
};
#endif

namespace TNL {

template< typename Index = int >
class tnlJPEGImage : public tnlImage< Index >
{
   public:
 
      typedef Index IndexType;
 
      tnlJPEGImage();
 
      bool openForRead( const String& fileName );
 
      template< typename Real,
                typename Device,
                typename Vector >
      bool read( const tnlRegionOfInterest< Index > roi,
                 const tnlGrid< 2, Real, Device, Index >& grid,
                 Vector& vector );
 
      template< typename Real,
                typename Device >
      bool openForWrite( const String& fileName,
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
      jpeg_decompress_struct decinfo;
      jpeg_compress_struct cinfo;
      int components;
      J_COLOR_SPACE color_space;
#endif
};

} // namespace TNL

#include <TNL/core/images/tnlJPEGImage_impl.h>

