#ifndef IMAGEINFOOBJ_H
#define IMAGEINFOOBJ_H

#include <dcmtk/dcmdata/dcfilefo.h>
#include <dcmtk/dcmdata/dcdeftag.h>

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

class ImageInfoObj
{
public:
    ImageInfoObj(DicomHeader &aDicomHeader);
    virtual ~ImageInfoObj();

public:
    ImagePositionToPatient getImagePositionToPatient();
    ImageOrientationToPatient getImageOrientationToPatient();
    double getSliceThickness();
    double getSliceLocation();
    PixelSpacing getPixelSpacing();
    int getNumberOfSlices();

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

#endif // IMAGEINFOOBJ_H
