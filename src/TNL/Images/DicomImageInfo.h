/***************************************************************************
                          DicomImageInfo.h  -  description
                             -------------------
    begin                : Jul 19, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
 
     Tomas Oberhuber     tomas.oberhuber@fjfi.cvut.cz
     Jiri Kafka          kafka9@seznam.cz
     Pavel Neskudla
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#ifdef HAVE_DCMTK_H
#define HAVE_CONFIG_H
#include <dcmtk/dcmdata/dcfilefo.h>
#include <dcmtk/dcmdata/dcdeftag.h>
#endif

namespace TNL {
namespace Images {   

class DicomHeader;

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

class DicomImageInfo
{
   public:
 
      inline DicomImageInfo( DicomHeader &DicomHeader);
 
      inline virtual ~DicomImageInfo();

      inline ImagePositionToPatient getImagePositionToPatient();
 
      inline ImageOrientationToPatient getImageOrientationToPatient();
 
      inline double getSliceThickness();
 
      inline double getSliceLocation();
 
      inline PixelSpacing getPixelSpacing();
 
      inline int getNumberOfSlices();

   private:
 
      DicomHeader &dicomHeader;
 
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

} // namespace Images
} // namespace TNL

#include <TNL/Images//DicomImageInfo_impl.h>

