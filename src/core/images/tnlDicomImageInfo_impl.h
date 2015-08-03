/***************************************************************************
                          tnlDicomImageInfo_impl.h  -  description
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

#include "tnlDicomImageInfo.h"
#include "tnlDicomHeader.h"

inline tnlDicomImageInfo::tnlDicomImageInfo( tnlDicomHeader& dicomHeader )
: dicomHeader( dicomHeader )
{
    isObjectRetrieved = false;
    depth = 0;
}

inline tnlDicomImageInfo::~tnlDicomImageInfo()
{
}

inline bool tnlDicomImageInfo::retrieveInfo()
{

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
   return 0;
}

inline ImagePositionToPatient tnlDicomImageInfo::getImagePositionToPatient()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return imagePositionToPatient;
}

inline ImageOrientationToPatient tnlDicomImageInfo::getImageOrientationToPatient()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return imageOrientationToPatient;
}

inline double tnlDicomImageInfo::getSliceThickness()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return sliceThickness;
}

inline double tnlDicomImageInfo::getSliceLocation()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return sliceLocation;
}

inline PixelSpacing tnlDicomImageInfo::getPixelSpacing()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return pixelSpacing;
}

inline int tnlDicomImageInfo::getNumberOfSlices()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return numberOfSlices;
}
