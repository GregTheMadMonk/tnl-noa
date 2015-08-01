#include <core/io/DicomSeries.h>
#include <core/io/SeriesInfoObj.h>
#include <dirent.h>


int findLastIndexOf(tnlString &str, char * const c)
{
    for (int i = str.getLength(); i > -1; i--)
    {
        char *a = &(str.operator [](i-1));
        if(*a == *c)
            return i;
    }
    return -1;
}

int filter(const struct dirent *dire)
{
    //check it is not DIR or unknowm d_type
    if(dire->d_type == DT_UNKNOWN && dire->d_type == DT_DIR)
        return 0;

    return 1;
}

inline DicomSeries::DicomSeries( const char* filePath)
{
    fileList = new tnlList<tnlString *>();
    dicomImage = 0;
    pixelData = 0;
    imagesInfo.imagesCount = 0;
    imagesInfo.maxColorValue = 0;
    imagesInfo.minColorValue = 128000;

    if(!loadDicomSeries(filePath))
        isLoaded = false;
    else
        isLoaded = true;
}

inline DicomSeries::~DicomSeries()
{
    fileList->DeepEraseAll();
    delete fileList;

    int length = dicomSeriesHeaders.getSize();
    for(int i = 0; i < length; i++)
    {
        tnlDicomHeader *header = dicomSeriesHeaders[i];
        delete header;
        header = 0;
    }

    if(dicomImage)
        delete dicomImage;

    if(pixelData)
        delete pixelData;
}

inline bool DicomSeries::retrieveFileList( const char *filePath)
{
    tnlString filePathString(filePath);
    tnlString suffix(filePath, filePathString.getLength() - 3);
    //char *ima = "ima";
    //char *dcm = "dcm";

    //check DICOM files
    if( suffix != "ima" && suffix != "dcm" )
        return false;

    int fileNamePosition = findLastIndexOf(filePathString,"/");

    //parse file path
    tnlString fileName(filePath, fileNamePosition);
    tnlString directoryPath(filePath, 0, filePathString.getLength() - fileNamePosition);

    int separatorPosition = findLastIndexOf(fileName, "_");
    if (separatorPosition == -1)
    {
        //try another separator
        separatorPosition = findLastIndexOf(fileName, "-");
    }
    if (separatorPosition == -1)
            return false;
    else
    {
        //numbered files
        tnlString fileNamePrefix(fileName.getString(), 0, fileName.getLength() - separatorPosition);

        struct dirent **dirp;
        tnlList<tnlString *> files;

        //scan and sort directory
        int ndirs = scandir(directoryPath.getString(), &dirp, filter, alphasort);
        for(int i = 0 ; i < ndirs; ++i)
        {
            files.Append(new tnlString((char *)dirp[i]->d_name));
            delete dirp[i];
        }

        for (int i = 0; i < files.getSize(); i++)
        {
            tnlString *file = new tnlString(files[i]->getString());

            //check if file prefix contained
            if (strstr(file->getString(), fileNamePrefix.getString()))
            {
                fileList->Append(new tnlString(directoryPath.operator +(*file)));
            }
            delete file;
            delete files[i];
        }
    }
    return true;
}

inline bool DicomSeries::loadImage(char *filePath, int number)
{
    //load header
    tnlDicomHeader *header = new tnlDicomHeader();
    dicomSeriesHeaders.setSize(fileList->getSize());
    dicomSeriesHeaders.setElement(number,header);
    if(!header->loadFromFile(filePath))
    {
        return false;
    }

    //check series UID
    const tnlString& seriesUID = dicomSeriesHeaders.operator [](0)->getSeriesInfoObj().getSeriesInstanceUID();
    if( seriesUID != header->getSeriesInfoObj().getSeriesInstanceUID() )
    {
        return false;
    }

    //load image
    if(dicomImage) delete dicomImage;
    dicomImage = NULL;

    dicomImage = new DicomImage(filePath);

    if(dicomImage->getFrameCount() > 1)
    {
        cout << filePath <<" not supported format-Dicom Image has more than one frame";
        return false;
    }

    if(!dicomImage->isMonochrome())
    {
        cout << filePath <<" not supported format--Dicom Image is not monochrome";
        return false;
    }

    if (dicomImage != NULL)
    {
        EI_Status imageStatus = dicomImage->getStatus();
        if (imageStatus == EIS_Normal)
        {
            //ok - image loaded
        }
        else if (EIS_MissingAttribute)
        {
            //bitmap is propably old ARC/NEMA format
            cerr << "Error: cannot load DICOM image(ACR/NEMA) (" << DicomImage::getString (dicomImage->getStatus()) << ")" << endl;

            delete dicomImage;
            dicomImage = NULL;
            return false;
        }
        else
        {
            delete dicomImage;
            dicomImage = NULL;
            cerr << "Error: cannot load DICOM image (" << DicomImage::getString (dicomImage->getStatus()) << ")" << endl;
            return false;
        }
    }

    if(number == 0)
    {
        imagesInfo.height = dicomImage->getHeight();
    }
    else if(dicomImage->getHeight() != imagesInfo.height)
    {
        cerr << filePath <<" image has bad height value\n";
    }

    if(number == 0)
    {
        imagesInfo.width = dicomImage->getWidth ();
    }
    else if(dicomImage->getWidth() != imagesInfo.width)
    {
        cerr << filePath <<" image has bad width value\n";
    }

    if(number == 0)
    {
        imagesInfo.bps = dicomImage->getDepth ();
    }
    else if(dicomImage->getDepth() != imagesInfo.bps)
    {
        cerr << filePath <<" image has bad bps value\n";
    }

    //update vales
    double min, max;
    dicomImage->getMinMaxValues(min, max);
    if(imagesInfo.minColorValue > min)
    {
        imagesInfo.minColorValue = min;
    }

    if(imagesInfo.maxColorValue < max)
    {
        imagesInfo.maxColorValue = max;
    }

    const unsigned long size = dicomImage->getOutputDataSize(16);
    //number of unsigned ints to allocate
    imagesInfo.frameUintsCount = size / sizeof(Uint16);
    if (number == 0)
    {//perform allocation only once
        imagesInfo.frameSize = size;
        if (pixelData)
            delete pixelData;
        pixelData = new Uint16[imagesInfo.frameUintsCount * fileList->getSize()];
    }
    else
    {//check image size for compatibility
        if (imagesInfo.frameSize != size)
        {
            cerr << filePath << " image has bad frame size value\n";
            return false;
        }
    }

    dicomImage->setMinMaxWindow();
    double center, width;
    dicomImage->getWindow(center,width);
    imagesInfo.window.center = center;
    imagesInfo.window.width = width ;
    dicomImage->setWindow(imagesInfo.window.center, imagesInfo.window.width);

    void *target = pixelData + (imagesInfo.frameUintsCount * imagesInfo.imagesCount);
    dicomImage->getOutputData(target,size,16);
    imagesInfo.imagesCount++;

    //delete image object - data are stored separatedly
    delete dicomImage;
    dicomImage = NULL;
    return true;
}


inline bool DicomSeries::loadDicomSeries( const char *filePath )
{
    //load list of files
    if(!retrieveFileList(filePath))
        return false;

    //load images
    int imagesCountToLoad = fileList->getSize();
    for (int i=0; i < imagesCountToLoad ;i++)
    {
         if(!loadImage((fileList->operator [](i))->getString(),i))
         {
             cerr << (fileList->operator [](i))->getString() << " skipped";
         }
     }
    return true;
}

inline int DicomSeries::getImagesCount()
{
    return imagesInfo.imagesCount;
}

inline const Uint16 *DicomSeries::getData()
{
    return pixelData;
}

inline int DicomSeries::getWidth()
{
    return imagesInfo.width;
}

inline int DicomSeries::getHeight()
{
    return imagesInfo.height;
}

inline int DicomSeries::getColorCount()
{
    return imagesInfo.colorsCount;
}

inline int DicomSeries::getBitsPerSampleCount()
{
    return imagesInfo.bps;
}

inline int DicomSeries::getMinColorValue()
{
    return imagesInfo.minColorValue;
}

inline WindowCenterWidth DicomSeries::getWindowDefaults()
{
    return imagesInfo.window;
}

inline int DicomSeries::getMaxColorValue()
{
    return imagesInfo.maxColorValue;
}

inline void DicomSeries::freeData()
{
    if (pixelData)
        delete pixelData;
    pixelData = NULL;
}

inline tnlDicomHeader &DicomSeries::getHeader(int image)
{
    //check user argument
    if((image > 0) | (image <= dicomSeriesHeaders.getSize()))
        return *dicomSeriesHeaders.getElement(image);
}

inline bool DicomSeries::isDicomSeriesLoaded()
{
    return isLoaded;
}

