#ifndef DICOMSERIE_H
#define DICOMSERIE_H

#define HAVE_CONFIG_H

#include "DicomHeader.h"

#include <dcmtk/config/osconfig.h>
#include <dcmtk/dcmimgle/dcmimage.h>

#include <core/arrays/tnlArray.h>

#include <core/tnlList.h>
#include <core/tnlString.h>

#include <dirent.h>
#include <string>
//#include <stdexcept>

template<> inline tnlString getParameterType< DicomHeader * > () { return tnlString( "DicomHeader *" ); }

/*class DicomSerieLoadException: public exception
{
public:
    virtual const char* what() const throw()
        {
        return "File could not be loaded";
        }
};*/

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

/***Class responsible for loading image data and headers of complete
    DICOM serie (searches the directory of the file). Call isDicomSerieLoaded()
    function to check if the load was successful.
  ***/
class DicomSeries
{
public:
    DicomSeries(char *filePath);
    virtual ~DicomSeries();

public:
    int getImagesCount();
    const Uint16 *getData();
    int getWidth();
    int getHeight();
    int getColorCount();
    int getBitsPerSampleCount();
    int getMinColorValue();
    WindowCenterWidth getWindowDefaults();
    int getMaxColorValue();
    void freeData();
    DicomHeader &getHeader(int image);
    bool isDicomSeriesLoaded();

private:
    bool loadDicomSeries(char *filePath);
    bool retrieveFileList(char *filePath);
    bool loadImage(char *filePath, int number);

    tnlList<tnlString *> *fileList;
    tnlArray<DicomHeader *,tnlHost,int> dicomSeriesHeaders;

    bool isLoaded;
    DicomImage *dicomImage;
    Uint16 *pixelData;
    ImagesInfo imagesInfo;
};

#endif // DICOMSERIES_H
