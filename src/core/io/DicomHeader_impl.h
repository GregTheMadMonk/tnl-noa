#include <core/io/DicomHeader.h>
#include <core/io/SeriesInfoObj.h>
#include <core/io/PatientInfoObj.h>
#include <core/io/ImageInfoObj.h>

inline tnlDicomHeader::tnlDicomHeader()
{
    fileFormat = new DcmFileFormat();
    isLoaded = false;
    imageInfoObj = new ImageInfoObj(*this);
    patientInfoObj = new PatientInfoObj(*this);
    seriesInfoObj = new SeriesInfoObj(*this);
}

inline tnlDicomHeader::~tnlDicomHeader()
{
    delete imageInfoObj;
    delete patientInfoObj;
    delete seriesInfoObj;
    delete fileFormat;
}

inline bool tnlDicomHeader::loadFromFile(const char *fileName)
{
    OFCondition status = fileFormat->loadFile(fileName);
    if(status.good())
    {
        isLoaded = true;
        return true;
    }
    isLoaded = false;
    return false;
}

inline DcmFileFormat &tnlDicomHeader::getFileFormat()
{
    return *fileFormat;
}

inline ImageInfoObj &tnlDicomHeader::getImageInfoObj()
{
    return *imageInfoObj;
}

inline PatientInfoObj &tnlDicomHeader::getPatientInfoObj()
{
    return *patientInfoObj;
}

inline SeriesInfoObj &tnlDicomHeader::getSeriesInfoObj()
{
    return *seriesInfoObj;
}

