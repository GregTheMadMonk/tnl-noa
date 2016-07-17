/***************************************************************************
                          tnlPNGImage.h  -  description
                             -------------------
    begin                : Jul 24, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <tnlConfig.h>

#ifdef HAVE_PNG_H
#include <png.h>
#endif

#include <core/tnlString.h>
#include <core/images/tnlImage.h>
#include <core/images/tnlRegionOfInterest.h>

namespace TNL {

template< typename Index = int >
class tnlPNGImage : public tnlImage< Index >
{
   public:
 
      typedef Index IndexType;
 
      tnlPNGImage();
 
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
 
      ~tnlPNGImage();
 
   protected:
 
      bool readHeader();
 
      template< typename Real,
                typename Device >
      bool writeHeader( const tnlGrid< 2, Real, Device, Index >& grid );
 
      FILE* file;

      bool fileOpen;

#ifdef HAVE_PNG_H
      png_structp png_ptr;

      png_infop info_ptr, end_info;
 
      png_byte color_type, bit_depth;
#endif
};

} // namespace TNL

#include <core/images/tnlPNGImage_impl.h>

