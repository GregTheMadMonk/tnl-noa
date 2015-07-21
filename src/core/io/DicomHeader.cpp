#include "DicomHeader.h"

DicomHeader::DicomHeader()
{
    fileFormat = new DcmFileFormat();
    isLoaded = false;
    imageInfoObj = new ImageInfoObj(*this);
    patientInfoObj = new PatientInfoObj(*this);
    seriesInfoObj = new SeriesInfoObj(*this);
}

DicomHeader::~DicomHeader()
{
    delete imageInfoObj;
    delete patientInfoObj;
    delete seriesInfoObj;
    delete fileFormat;
}

bool DicomHeader::loadFromFile(const char *fileName)
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

DcmFileFormat &DicomHeader::getFileFormat()
{
    return *fileFormat;
}

ImageInfoObj &DicomHeader::getImageInfoObj()
{
    return *imageInfoObj;
}

PatientInfoObj &DicomHeader::getPatientInfoObj()
{
    return *patientInfoObj;
}
SeriesInfoObj &DicomHeader::getSeriesInfoObj()
{
    return *seriesInfoObj;
}

