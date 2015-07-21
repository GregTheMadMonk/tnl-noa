#ifndef SERIESINFOOBJ_H
#define SERIESINFOOBJ_H

#include <dcmtk/dcmdata/dcfilefo.h>
#include <dcmtk/dcmdata/dcdeftag.h>

class DicomHeader;

/***
 * SeriesInfoObj class stores selected informations about DICOM series.
 * (accesses information via DicomHeader class)
 */
class SeriesInfoObj
{
public:
    SeriesInfoObj(DicomHeader &aDicomHeader);
    virtual ~SeriesInfoObj();

public:
    const char *getModality();
    const char *getStudyInstanceUID();
    const char *getSeriesInstanceUID();
    const char *getSeriesDescription();
    const char *getSeriesNumber();
    const char *getSeriesDate();
    const char *getSeriesTime();
    const char *getPerformingPhysiciansName();
    const char *getPerformingPhysicianIdentificationSequence();
    const char *getOperatorsName();
    const char *getOperatorIdentificationSequence();
    const char *getAcquisitionTime();

private:
    DicomHeader &dicomHeader;
    bool retrieveInfo();
    bool isObjectRetrieved;

    OFString modality;
    OFString studyInstanceUID;
    OFString seriesInstanceUID;
    OFString seriesNumber;
    OFString seriesDescription;
    OFString seriesDate;
    OFString seriesTime;
    OFString performingPhysiciansName;
    OFString performingPhysicianIdentificationSequence;
    OFString operatorsName;
    OFString operatorIdentificationSequence;
    OFString acquisitionTime;
};

#endif // SERIESINFOOBJ_H
