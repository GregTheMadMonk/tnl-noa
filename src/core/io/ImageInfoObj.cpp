#include "ImageInfoObj.h"
#include "DicomHeader.h"

ImageInfoObj::ImageInfoObj(DicomHeader &aDicomHeader) : dicomHeader(aDicomHeader)
{
    isObjectRetrieved = false;
    depth = 0;
}

ImageInfoObj::~ImageInfoObj()
{
}

bool ImageInfoObj::retrieveInfo()
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

ImagePositionToPatient ImageInfoObj::getImagePositionToPatient()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return imagePositionToPatient;
}

ImageOrientationToPatient ImageInfoObj::getImageOrientationToPatient()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return imageOrientationToPatient;
}

double ImageInfoObj::getSliceThickness()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return sliceThickness;
}
double ImageInfoObj::getSliceLocation()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return sliceLocation;
}

PixelSpacing ImageInfoObj::getPixelSpacing()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return pixelSpacing;
}

int ImageInfoObj::getNumberOfSlices()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return numberOfSlices;
}
