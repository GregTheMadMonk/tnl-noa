/***************************************************************************
                          JPEGImage.h  -  description
                             -------------------
    begin                : Jul 25, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#ifdef HAVE_JPEG_H
#include <jpeglib.h>
#endif

#include <TNL/String.h>
#include <TNL/Images/Image.h>
#include <TNL/Images/RegionOfInterest.h>

#ifdef HAVE_JPEG_H
struct my_error_mgr
{
   jpeg_error_mgr pub;
   jmp_buf setjmp_buffer;
};
#endif

namespace TNL {
namespace Images {   

template< typename Index = int >
class JPEGImage : public Image< Index >
{
   public:
 
      typedef Index IndexType;
 
      JPEGImage();
 
      bool openForRead( const String& fileName );
 
      template< typename MeshReal,
                typename Device,
                typename Real >
      bool read( const RegionOfInterest< Index > roi,
                 Functions::MeshFunction< Meshes::Grid< 2, MeshReal, Device, Index >, 2, Real >& function );
      
 
      template< typename Real,
                typename Device >
      bool openForWrite( const String& fileName,
                         Meshes::Grid< 2, Real, Device, Index >& grid );
 
      // TODO: Obsolete
      template< typename Real,
                typename Device,
                typename Vector >
      bool write( const Meshes::Grid< 2, Real, Device, Index >& grid,
                  Vector& vector );
      
      template< typename MeshReal,
                typename Device,
                typename Real >
      bool write( const Functions::MeshFunction< Meshes::Grid< 2, MeshReal, Device, Index >, 2, Real >& function );
      
 
      void close();
 
      ~JPEGImage();
 
   protected:
 
      bool readHeader();
 
      template< typename Real,
                typename Device >
      bool writeHeader( const Meshes::Grid< 2, Real, Device, Index >& grid );
 
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

} // namespace Images
} // namespace TNL

#include <TNL/Images//JPEGImage_impl.h>

