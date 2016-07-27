/***************************************************************************
                          tnlPGMImage.h  -  description
                             -------------------
    begin                : Jul 20, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

#include <TNL/String.h>
#include <TNL/Images//Image.h>
#include <TNL/Images//RegionOfInterest.h>
#include <fstream>

namespace TNL {
namespace Images {   

template< typename Index = int >
class tnlPGMImage : public tnlImage< Index >
{
   public:
 
      typedef Index IndexType;
 
      tnlPGMImage();
 
      bool openForRead( const String& fileName );
 
      template< typename Real,
                typename Device,
                typename Vector >
      bool read( const RegionOfInterest< Index > roi,
                 const tnlGrid< 2, Real, Device, Index >& grid,
                 Vector& vector );
 
      template< typename Real,
                typename Device >
      bool openForWrite( const String& fileName,
                         tnlGrid< 2, Real, Device, Index >& grid,
                         bool binary = true );
 
      template< typename Real,
                typename Device,
                typename Vector >
      bool write( const tnlGrid< 2, Real, Device, Index >& grid,
                  Vector& vector );

 
      void close();
 
      ~tnlPGMImage();
 
      protected:
 
         bool readHeader();
 
         template< typename Real,
                   typename Device >
         bool writeHeader( const tnlGrid< 2, Real, Device, Index >& grid,
                           bool binary );
 
         bool binary;
 
         IndexType maxColors;
 
         std::fstream file;
 
         bool fileOpen;
};

} // namespace Images
} // namespace TNL

#include <TNL/Images//PGMImage_impl.h>


