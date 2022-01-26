// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <list>

#include <TNL/Containers/Array.h>
#include <TNL/String.h>
#include <TNL/TypeInfo.h>
#include <TNL/Meshes/Grid.h>

#include <TNL/Images/DicomHeader.h>
#include <TNL/Images/Image.h>
#include <TNL/Images/RegionOfInterest.h>

#ifdef HAVE_DCMTK_H
   #define USING_STD_NAMESPACE
   #include <dcmtk/config/osconfig.h>
   #include <dcmtk/dcmimgle/dcmimage.h>
#endif

#include <dirent.h>
#include <string>

namespace TNL {
namespace Images {

struct WindowCenterWidth
{
   float center;
   float width;
};

struct ImagesInfo
{
   int imagesCount, frameUintsCount, bps, colorsCount, mainFrameIndex, frameSize, maxColorValue, minColorValue;
   WindowCenterWidth window;
};

/***
 * Class responsible for loading image data and headers of complete
 * DICOM serie (searches the directory of the file). Call isDicomSeriesLoaded()
 * function to check if the load was successful.
 */
class DicomSeries : public Image< int >
{
public:
   typedef int IndexType;

   inline DicomSeries( const String& filePath );

   inline virtual ~DicomSeries();

   inline int
   getImagesCount();

   template< typename Real, typename Device, typename Index, typename Vector >
   bool
   getImage( const int imageIdx,
             const Meshes::Grid< 2, Real, Device, Index >& grid,
             const RegionOfInterest< int > roi,
             Vector& vector );

#ifdef HAVE_DCMTK_H
   inline const Uint16*
   getData( int imageNumber = 0 );
#endif

   inline int
   getColorCount();

   inline int
   getBitsPerSampleCount();

   inline int
   getMinColorValue();

   inline WindowCenterWidth
   getWindowDefaults();

   inline int
   getMaxColorValue();

   inline void
   freeData();

   inline DicomHeader&
   getHeader( int image );

   inline bool
   isDicomSeriesLoaded();

private:
   bool
   loadDicomSeries( const String& filePath );

   bool
   retrieveFileList( const String& filePath );

   bool
   loadImage( const String& filePath, int number );

   std::list< String > fileList;

   Containers::Array< DicomHeader*, Devices::Host, int > dicomSeriesHeaders;

   bool isLoaded;

#ifdef HAVE_DCMTK_H
   DicomImage* dicomImage;

   Uint16* pixelData;
#endif

   ImagesInfo imagesInfo;
};

}  // namespace Images
}  // namespace TNL

#include <TNL/Images/DicomSeries_impl.h>
