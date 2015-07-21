#include "PatientInfoObj.h"
#include "DicomHeader.h"

PatientInfoObj::PatientInfoObj(DicomHeader &aDicomHeader) : dicomHeader(aDicomHeader)
{
    isObjectRetrieved = false;
}

PatientInfoObj::~PatientInfoObj()
{
}

bool PatientInfoObj::retrieveInfo()
{
    name = "";
    dicomHeader.getFileFormat().getDataset()->findAndGetOFString(DCM_PatientName, name);
    sex = "";
    dicomHeader.getFileFormat().getDataset()->findAndGetOFString(DCM_PatientSex, sex);
    ID = "";
    dicomHeader.getFileFormat().getDataset()->findAndGetOFString(DCM_PatientID, ID);
    weight = "";
    dicomHeader.getFileFormat().getDataset()->findAndGetOFString(DCM_PatientWeight, weight);
    patientPosition = "";
    dicomHeader.getFileFormat().getDataset()->findAndGetOFString(DCM_PatientPosition, patientPosition);
    patientOrientation = "";
    dicomHeader.getFileFormat().getDataset()->findAndGetOFString(DCM_PatientOrientation, patientOrientation);

    isObjectRetrieved = true;
    return 0;
}

const char *PatientInfoObj::getName()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return name.c_str();
}

const char *PatientInfoObj::getSex()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return sex.c_str();
}

const char *PatientInfoObj::getID()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return ID.c_str();
}

const char *PatientInfoObj::getWeight()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return weight.c_str();
}

const char *PatientInfoObj::getPosition()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return patientPosition.c_str();
}

const char *PatientInfoObj::getOrientation()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return patientOrientation.c_str();
}
