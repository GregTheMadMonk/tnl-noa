/***************************************************************************
                          DicomImageInfo_impl.h  -  description
                             -------------------
    begin                : Jul 19, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
 
     Tomas Oberhuber     tomas.oberhuber@fjfi.cvut.cz
     Jiri Kafka          kafka9@seznam.cz
     Pavel Neskudla
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Images//DicomImageInfo.h>
#include <TNL/Images//DicomHeader.h>

namespace TNL {
namespace Images {   

inline DicomImageInfo::DicomImageInfo( DicomHeader& dicomHeader )
: dicomHeader( dicomHeader )
{
    isObjectRetrieved = false;
    depth = 0;
}

inline DicomImageInfo::~DicomImageInfo()
{
}

inline bool DicomImageInfo::retrieveInfo()
{
#ifdef HAVE_DCMTK_H
   dicomHeader.getFileFormat().getDataset()->findAndGetFloat64(DCM_ImagePositionPatient,imagePositionToPatient.x,0);
   dicomHeader.getFileFormat().getDataset()->findAndGetFloat64(DCM_ImagePositionPatient,imagePositionToPatient.y,1);
   dicomHeader.getFileFormat().getDataset()->findAndGetFloat64(DCM_ImagePositionPatient,imagePositionToPatient.z,2);

   dicomHeader.getFileFormat().getDataset()->findAndGetFloat64(DCM_ImageOrientationPatient,imageOrientationToPatient.row.x,0);
   dicomHeader.getFileFormat().getDataset()->findAndGetFloat64(DCM_ImageOrientationPatient,imageOrientationToPatient.row.y,1);
   dicomHeader.getFileFormat().getDataset()->findAndGetFloat64(DCM_ImageOrientationPatient,imageOrientationToPatient.row.z,2);
   dicomHeader.getFileFormat().getDataset()->findAndGetFloat64(DCM_ImageOrientationPatient,imageOrientationToPatient.column.x,3);
   dicomHeader.getFileFormat().getDataset()->findAndGetFloat64(DCM_ImageOrientationPatient,imageOrientationToPatient.column.y,4);
   dicomHeader.getFileFormat().getDataset()->findAndGetFloat64(DCM_ImageOrientationPatient,imageOrientationToPatient.column.z,5);

   dicomHeader.getFileFormat().getDataset()->findAndGetFloat64(DCM_SliceThickness,sliceThickness);
   dicomHeader.getFileFormat().getDataset()->findAndGetFloat64(DCM_SliceLocation,sliceLocation);

   Uint16 slicesCount;
   dicomHeader.getFileFormat().getDataset()->findAndGetUint16 (DCM_NumberOfSlices, slicesCount);
   numberOfSlices = slicesCount;

   dicomHeader.getFileFormat().getDataset()->findAndGetFloat64(DCM_PixelSpacing,pixelSpacing.x,0);
   dicomHeader.getFileFormat().getDataset()->findAndGetFloat64(DCM_PixelSpacing,pixelSpacing.y,1);

   isObjectRetrieved = true;
   return true;
#else
   std::cerr << "DICOM format is not supported in this build of TNL." << std::endl;
   return false;
#endif
}

inline ImagePositionToPatient DicomImageInfo::getImagePositionToPatient()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return imagePositionToPatient;
}

inline ImageOrientationToPatient DicomImageInfo::getImageOrientationToPatient()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return imageOrientationToPatient;
}

inline double DicomImageInfo::getSliceThickness()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return sliceThickness;
}

inline double DicomImageInfo::getSliceLocation()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return sliceLocation;
}

inline PixelSpacing DicomImageInfo::getPixelSpacing()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return pixelSpacing;
}

inline int DicomImageInfo::getNumberOfSlices()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return numberOfSlices;
}

} // namespace Images
} // namespace TNL