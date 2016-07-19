/***************************************************************************
                          tnlDicomSeries.h  -  description
                             -------------------
    begin                : Jul 31, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
 
     Tomas Oberhuber     tomas.oberhuber@fjfi.cvut.cz
     Jiri Kafka          kafka9@seznam.cz
     Pavel Neskudla
 
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/core/arrays/tnlArray.h>
#include <TNL/core/tnlList.h>
#include <TNL/core/tnlString.h>
#include <TNL/core/param-types.h>
#include <TNL/core/images/tnlImage.h>
#include <TNL/core/images/tnlDicomHeader.h>
#include <TNL/core/images/tnlRegionOfInterest.h>
#include <TNL/mesh/tnlGrid.h>
#include <TNL/tnlConfig.h>

#ifdef HAVE_DCMTK_H
#define USING_STD_NAMESPACE
#include <dcmtk/config/osconfig.h>
#include <dcmtk/dcmimgle/dcmimage.h>
#endif

#include <dirent.h>
#include <string>

namespace TNL {

template<> inline tnlString getType< tnlDicomHeader * > () { return tnlString( "tnlDicomHeader *" ); }

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
class tnlDicomSeries : public tnlImage< int >
{
   public:
 
      inline tnlDicomSeries( const tnlString& filePath );
 
      inline virtual ~tnlDicomSeries();

      inline int getImagesCount();
 
      template< typename Real,
                typename Device,
                typename Index,
                typename Vector >
      bool getImage( const int imageIdx,
                     const tnlGrid< 2, Real, Device, Index >& grid,
                     const tnlRegionOfInterest< int > roi,
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
 
      inline tnlDicomHeader &getHeader(int image);
 
      inline bool isDicomSeriesLoaded();

   private:
 
      bool loadDicomSeries( const tnlString& filePath );
 
      bool retrieveFileList( const tnlString& filePath );
 
      bool loadImage( const tnlString& filePath, int number );

      tnlList< tnlString > fileList;
 
      tnlArray<tnlDicomHeader *,tnlHost,int> dicomSeriesHeaders;

      bool isLoaded;
 
#ifdef HAVE_DCMTK_H
      DicomImage *dicomImage;
 
      Uint16 *pixelData;
#endif
 
      ImagesInfo imagesInfo;
};

} // namespace TNL

#include <TNL/core/images/tnlDicomSeries_impl.h>

