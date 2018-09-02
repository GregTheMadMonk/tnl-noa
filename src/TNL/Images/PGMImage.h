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
class PGMImage : public Image< Index >
{
   public:
 
      typedef Index IndexType;
 
      PGMImage();
 
      bool openForRead( const String& fileName );
 
      template< typename MeshReal,
                typename Device,
                typename Real >
      bool read( const RegionOfInterest< Index > roi,
                 Functions::MeshFunction< Meshes::Grid< 2, MeshReal, Device, Index >, 2, Real >& function );
 
      template< typename Real,
                typename Device >
      bool openForWrite( const String& fileName,
                         Meshes::Grid< 2, Real, Device, Index >& grid,
                         bool binary = true );

      // TODO: obsolete
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
 
      ~PGMImage();
 
      protected:
 
         bool readHeader();
 
         template< typename Real,
                   typename Device >
         bool writeHeader( const Meshes::Grid< 2, Real, Device, Index >& grid,
                           bool binary );
 
         bool binary;
 
         IndexType maxColors;
 
         std::fstream file;
 
         bool fileOpen;
};

} // namespace Images
} // namespace TNL

#include <TNL/Images//PGMImage_impl.h>


