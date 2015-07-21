#ifndef PATIENTINFOOBJ_H
#define PATIENTINFOOBJ_H

class DicomHeader;

#include <dcmtk/dcmdata/dcfilefo.h>
#include <dcmtk/dcmdata/dcdeftag.h>

/***PatientInfoObj class stores selected informations about patient.
  (accesses information via DicomHeader class)
  ***/
class PatientInfoObj
{
public:
    PatientInfoObj(DicomHeader &aDicomHeader);
    virtual ~PatientInfoObj();

public:
    const char *getName();
    const char *getSex();
    const char *getID();
    const char *getWeight();
    const char *getPosition();
    const char *getOrientation();

private:

    DicomHeader &dicomHeader;
    bool retrieveInfo();
    bool isObjectRetrieved;

    OFString name;
    OFString sex;
    OFString ID;
    OFString weight;
    OFString patientPosition;
    OFString patientOrientation;
};

#endif // PATIENTINFOOBJ_H
