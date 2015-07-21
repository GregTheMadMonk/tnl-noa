#include "SeriesInfoObj.h"
#include "DicomHeader.h"
#include <stdio.h>

SeriesInfoObj::SeriesInfoObj(DicomHeader &aDicomHeader) : dicomHeader(aDicomHeader)
{
    isObjectRetrieved = false;
}

SeriesInfoObj::~SeriesInfoObj()
{
}

bool SeriesInfoObj::retrieveInfo()
{
    modality = "";
    dicomHeader.getFileFormat().getDataset()->findAndGetOFString(DCM_Modality, modality);
    studyInstanceUID = "";
    dicomHeader.getFileFormat().getDataset()->findAndGetOFString(DCM_StudyInstanceUID, studyInstanceUID);
    seriesInstanceUID = "";
    dicomHeader.getFileFormat().getDataset()->findAndGetOFString(DCM_SeriesInstanceUID, seriesInstanceUID);
    seriesNumber = "";
    dicomHeader.getFileFormat().getDataset()->findAndGetOFString(DCM_SeriesNumber, seriesNumber);
    seriesDescription = "";
    dicomHeader.getFileFormat().getDataset()->findAndGetOFString(DCM_SeriesDescription, seriesDescription);
    seriesDate = "";
    dicomHeader.getFileFormat().getDataset()->findAndGetOFString(DCM_SeriesDate, seriesDate);
    seriesTime = "";
    dicomHeader.getFileFormat().getDataset()->findAndGetOFString(DCM_SeriesTime, seriesTime);
    performingPhysiciansName = "";
    dicomHeader.getFileFormat().getDataset()->findAndGetOFString(DCM_PerformingPhysicianName, performingPhysiciansName);
    performingPhysicianIdentificationSequence = "";
    dicomHeader.getFileFormat().getDataset()->findAndGetOFString(DCM_PerformingPhysicianIdentificationSequence, performingPhysicianIdentificationSequence);
    operatorsName = "";
    dicomHeader.getFileFormat().getDataset()->findAndGetOFString(DCM_OperatorsName, operatorsName);
    operatorIdentificationSequence = "";
    dicomHeader.getFileFormat().getDataset()->findAndGetOFString(DCM_OperatorIdentificationSequence, operatorIdentificationSequence);
    OFString frameTime = "";
    dicomHeader.getFileFormat().getDataset()->findAndGetOFString(DCM_FrameAcquisitionDuration, frameTime);
    OFString faDateTime = "";
    dicomHeader.getFileFormat().getDataset()->findAndGetOFString(DCM_FrameAcquisitionDateTime, faDateTime);
    OFString faRefTime = "";
    dicomHeader.getFileFormat().getDataset()->findAndGetOFString(DCM_FrameReferenceTime, faRefTime);
    OFString AFD = "";
    dicomHeader.getFileFormat().getDataset()->findAndGetOFString(DCM_ActualFrameDuration, AFD);
    acquisitionTime = "";
    dicomHeader.getFileFormat().getDataset()->findAndGetOFString(DCM_AcquisitionTime, acquisitionTime);

    //prostudovat delay time
    //OFString delayTime = "";
    //dicomHeader.getFileFormat().getDataset()->findAndGetOFString(DCM_DelayTime, delayTime);

    //std::cout << faDateTime << " " << faRefTime << " "<< AFD << " " << AT << std::endl;

    isObjectRetrieved = true;
    return 0;
}

const char *SeriesInfoObj::getModality()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return modality.c_str();
}

const char *SeriesInfoObj::getStudyInstanceUID()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return studyInstanceUID.c_str();
}

const char *SeriesInfoObj::getSeriesInstanceUID()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return seriesInstanceUID.c_str();
}

const char *SeriesInfoObj::getSeriesNumber()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return seriesNumber.c_str();
}

const char *SeriesInfoObj::getSeriesDescription()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return seriesDescription.c_str();
}

const char *SeriesInfoObj::getSeriesDate()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return seriesDate.c_str();
}

const char *SeriesInfoObj::getSeriesTime()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return seriesTime.c_str();
}

const char *SeriesInfoObj::getPerformingPhysiciansName()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return performingPhysiciansName.c_str();
}

const char *SeriesInfoObj::getPerformingPhysicianIdentificationSequence()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return performingPhysicianIdentificationSequence.c_str();
}

const char *SeriesInfoObj::getOperatorsName()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return operatorsName.c_str();
}

const char *SeriesInfoObj::getOperatorIdentificationSequence()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return operatorIdentificationSequence.c_str();
}

const char *SeriesInfoObj::getAcquisitionTime()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return acquisitionTime.c_str();
}
