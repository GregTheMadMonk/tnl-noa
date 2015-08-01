#ifndef IMAGEINFOOBJ_H
#define IMAGEINFOOBJ_H

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

class ImageInfoObj
{
   public:
      
      inline ImageInfoObj( tnlDicomHeader &tnlDicomHeader);
       
      inline virtual ~ImageInfoObj();

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

#include <core/io/ImageInfoObj_impl.h>

#endif // IMAGEINFOOBJ_H
