/***************************************************************************
                          DicomSeries.h  -  description
                             -------------------
    begin                : Jul 31, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
 
     Tomas Oberhuber     tomas.oberhuber@fjfi.cvut.cz
     Jiri Kafka          kafka9@seznam.cz
     Pavel Neskudla
 
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Arrays/Array.h>
#include <TNL/List.h>
#include <TNL/String.h>
#include <TNL/core/param-types.h>
#include <TNL/Images//Image.h>
#include <TNL/Images//DicomHeader.h>
#include <TNL/Images//RegionOfInterest.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/tnlConfig.h>

#ifdef HAVE_DCMTK_H
#define USING_STD_NAMESPACE
#include <dcmtk/config/osconfig.h>
#include <dcmtk/dcmimgle/dcmimage.h>
#endif

#include <dirent.h>
#include <string>

namespace TNL {

template<> inline String getType< Images::DicomHeader * > () { return String( "DicomHeader *" ); }

namespace Images {   

struct WindowCenterWidth
{
    float center;
    float width;
};

struct ImagesInfo
{
    int imagesCount, frameUintsCount, bps, colorsCount, mainFrameIndex,
        frameSize, maxColorValue, minColorValue;
    WindowCenterWidth window;
};

/***
 * Class responsible for loading image data and headers of complete
 * DICOM serie (searches the directory of the file). Call isDicomSeriesLoaded()
 * function to check if the load was successful.
 */
class DicomSeries : public tnlImage< int >
{
   public:
 
      inline DicomSeries( const String& filePath );
 
      inline virtual ~DicomSeries();

      inline int getImagesCount();
 
      template< typename Real,
                typename Device,
                typename Index,
                typename Vector >
      bool getImage( const int imageIdx,
                     const Meshes::Grid< 2, Real, Device, Index >& grid,
                     const RegionOfInterest< int > roi,
                     Vector& vector );
 
#ifdef HAVE_DCMTK_H
      inline const Uint16 *getData( int imageNumber = 0 );
#endif
 
      inline int getColorCount();
 
      inline int getBitsPerSampleCount();
 
      inline int getMinColorValue();
 
      inline WindowCenterWidth getWindowDefaults();
 
      inline int getMaxColorValue();
 
      inline void freeData();
 
      inline DicomHeader &getHeader(int image);
 
      inline bool isDicomSeriesLoaded();

   private:
 
      bool loadDicomSeries( const String& filePath );
 
      bool retrieveFileList( const String& filePath );
 
      bool loadImage( const String& filePath, int number );

      List< String > fileList;
 
      Arrays::Array<DicomHeader *,Devices::Host,int> dicomSeriesHeaders;

      bool isLoaded;
 
#ifdef HAVE_DCMTK_H
      DicomImage *dicomImage;
 
      Uint16 *pixelData;
#endif
 
      ImagesInfo imagesInfo;
};

} // namespace Images
} // namespace TNL

#include <TNL/Images//DicomSeries_impl.h>

