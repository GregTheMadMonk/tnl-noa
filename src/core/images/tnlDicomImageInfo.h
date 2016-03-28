/***************************************************************************
                          tnlDicomImageInfo.h  -  description
                             -------------------
    begin                : Jul 19, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.                                       
     
     Tomas Oberhuber     tomas.oberhuber@fjfi.cvut.cz
     Jiri Kafka          kafka9@seznam.cz
     Pavel Neskudla
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLDICOMIMAGEINFO_H
#define TNLDICOMIMAGEINFO_H

#include <tnlConfig.h>

#ifdef HAVE_DCMTK_H
#define HAVE_CONFIG_H
#include <dcmtk/dcmdata/dcfilefo.h>
#include <dcmtk/dcmdata/dcdeftag.h>
#endif

class tnlDicomHeader;

/***ImageInfoObj class stores selected informations about images.
  (accesses information via DicomHeader class)
  ***/
struct ImagePositionToPatient
    {
    double x, y,z;
    };

struct DirectionCosines
    {
    double x, y, z;
    };

struct ImageOrientationToPatient
    {
    DirectionCosines row;
    DirectionCosines column;
    };

struct PixelSpacing
    {
    double x, y;
    };

class tnlDicomImageInfo
{
   public:
      
      inline tnlDicomImageInfo( tnlDicomHeader &tnlDicomHeader);
       
      inline virtual ~tnlDicomImageInfo();

      inline ImagePositionToPatient getImagePositionToPatient();
      
      inline ImageOrientationToPatient getImageOrientationToPatient();
       
      inline double getSliceThickness();
       
      inline double getSliceLocation();
       
      inline PixelSpacing getPixelSpacing();
       
      inline int getNumberOfSlices();

   private:
      
      tnlDicomHeader &dicomHeader;
       
      bool retrieveInfo();
       
      bool isObjectRetrieved;

      double sliceLocation;
       
      double sliceThickness;
       
      ImagePositionToPatient imagePositionToPatient;
       
      ImageOrientationToPatient imageOrientationToPatient;
              
      PixelSpacing pixelSpacing;
       
      int numberOfSlices;
       
      int width, height, depth;
};

#include <core/images/tnlDicomImageInfo_impl.h>

#endif // TNLDICOMIMAGEINFO_H