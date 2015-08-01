/***************************************************************************
                          tnlDicomSeries.h  -  description
                             -------------------
    begin                : Jul 31, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.

   Tomas Oberhuber  tomas.oberhuber@fjfi.cvut.cz
   Jiri Kafka       kafka9@seznam.cz

 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLDICOMSERIES_H
#define TNLDICOMSERIES_H

#include <core/arrays/tnlArray.h>
#include <core/tnlList.h>
#include <core/tnlString.h>
#include <core/param-types.h>
#include <core/io/DicomHeader.h>
#include <tnlConfig.h>


#ifdef HAVE_DCMTK_H
#define USING_STD_NAMESPACE
#include <dcmtk/config/osconfig.h>
#include <dcmtk/dcmimgle/dcmimage.h>
#endif

#include <dirent.h>
#include <string>

template<> inline tnlString getType< tnlDicomHeader * > () { return tnlString( "tnlDicomHeader *" ); }

struct WindowCenterWidth
{
    float center;
    float width;
};

struct ImagesInfo
{
    int width, height, imagesCount, frameUintsCount, bps, colorsCount, mainFrameIndex,
        frameSize, maxColorValue, minColorValue;
    WindowCenterWidth window;
};

/***
 * Class responsible for loading image data and headers of complete
 * DICOM serie (searches the directory of the file). Call isDicomSeriesLoaded()
 * function to check if the load was successful.
 */
class DicomSeries
{
   public:
      
      inline DicomSeries( const char *filePath );
       
      inline virtual ~DicomSeries();

      inline int getImagesCount();
       
#ifdef HAVE_DCMTK_H       
      inline const Uint16 *getData();
#endif       
       
      inline int getWidth();
       
      inline int getHeight();
       
      inline int getColorCount();
       
      inline int getBitsPerSampleCount();
       
      inline int getMinColorValue();
       
      inline WindowCenterWidth getWindowDefaults();
       
      inline int getMaxColorValue();
       
      inline void freeData();
       
      inline tnlDicomHeader &getHeader(int image);
       
      inline bool isDicomSeriesLoaded();

   private:
      
      bool loadDicomSeries( const char *filePath );
       
      bool retrieveFileList( const char *filePath );
       
      bool loadImage( char *filePath, int number );

      tnlList<tnlString *> *fileList;
       
      tnlArray<tnlDicomHeader *,tnlHost,int> dicomSeriesHeaders;

      bool isLoaded;
      
#ifdef HAVE_DCMTK_H       
      DicomImage *dicomImage;
       
      Uint16 *pixelData;
#endif       
       
      ImagesInfo imagesInfo;
};

#include <core/io/DicomSeries_impl.h>

#endif // TNLDICOMSERIES_H
