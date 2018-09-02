/***************************************************************************
                          PNGImage.h  -  description
                             -------------------
    begin                : Jul 24, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/tnlConfig.h>

#ifdef HAVE_PNG_H
#include <png.h>
#endif

#include <TNL/String.h>
#include <TNL/Images/Image.h>
#include <TNL/Images/RegionOfInterest.h>
#include <TNL/Functions/MeshFunction.h>

namespace TNL {
namespace Images {   

template< typename Index = int >
class PNGImage : public Image< Index >
{
   public:
 
      using IndexType = Index;
 
      PNGImage();
 
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
 
      ~PNGImage();
 
   protected:
 
      bool readHeader();
 
      template< typename Real,
                typename Device >
      bool writeHeader( const Meshes::Grid< 2, Real, Device, Index >& grid );
 
      FILE* file;

      bool fileOpen;

#ifdef HAVE_PNG_H
      png_structp png_ptr;

      png_infop info_ptr, end_info;
 
      png_byte color_type, bit_depth;
#endif
};

} // namespace Images
} // namespace TNL

#include <TNL/Images//PNGImage_impl.h>

