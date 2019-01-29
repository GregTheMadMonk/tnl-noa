/***************************************************************************
                          DicomSeriesInfo_impl.h  -  description
                             -------------------
    begin                : Jul 19, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
 
     Tomas Oberhuber     tomas.oberhuber@fjfi.cvut.cz
     Jiri Kafka          kafka9@seznam.cz
     Pavel Neskudla
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Images//DicomSeriesInfo.h>
#include <TNL/Images//DicomHeader.h>
#include <stdio.h>

namespace TNL {
namespace Images {   

inline DicomSeriesInfo::DicomSeriesInfo( DicomHeader &dicomHeader)
: dicomHeader( dicomHeader )
{
    isObjectRetrieved = false;
}

inline DicomSeriesInfo::~DicomSeriesInfo()
{
}

inline bool DicomSeriesInfo::retrieveInfo()
{
#ifdef HAVE_DCMTK_H
   OFString str;
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_Modality, str );
   this->modality = str.data();
 
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_StudyInstanceUID, str );
   this->studyInstanceUID = str.data();
 
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_SeriesInstanceUID, str );
   this->seriesInstanceUID = str.data();
 
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_SeriesNumber, str );
   this->seriesNumber = str.data();
 
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_SeriesDescription, str );
   this->seriesDescription = str.data();
 
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_SeriesDate, str );
   this->seriesDate = str.data();
 
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_SeriesTime, str );
   this->seriesTime = str.data();
 
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_PerformingPhysicianName, str );
   this->performingPhysiciansName = str.data();
 
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_PerformingPhysicianIdentificationSequence, str );
   this->performingPhysicianIdentificationSequence = str.data();
 
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_OperatorsName, str );
   this->operatorsName = str.data();
 
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_OperatorIdentificationSequence, str );
   this->operatorIdentificationSequence = str.data();
 
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_FrameAcquisitionDuration, str );
   this->frameTime = str.data();
 
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_FrameAcquisitionDateTime, str );
   this->faDateTime = str.data();
 
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_FrameReferenceTime, str );
   this->faRefTime = str.data();
 
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_ActualFrameDuration, str );
   this->AFD = str.data();
 
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_AcquisitionTime, str );
   this->acquisitionTime = str.data();

    //prostudovat delay time
    //OFString delayTime = "";
    //dicomHeader.getFileFormat().getDataset()->findAndGetOFString(DCM_DelayTime, delayTime);

    //std::cout << faDateTime << " " << faRefTime << " "<< AFD << " " << AT << std::endl;

    isObjectRetrieved = true;
    return true;
#else
    std::cerr << "DICOM format is not supported in this build of TNL." << std::endl;
    return false;
#endif
}

inline const String& DicomSeriesInfo::getModality()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return this->modality;
}

inline const String& DicomSeriesInfo::getStudyInstanceUID()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return this->studyInstanceUID;
}

inline const String& DicomSeriesInfo::getSeriesInstanceUID()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return this->seriesInstanceUID;
}

inline const String& DicomSeriesInfo::getSeriesNumber()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return this->seriesNumber;
}

inline const String& DicomSeriesInfo::getSeriesDescription()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return this->seriesDescription;
}

inline const String& DicomSeriesInfo::getSeriesDate()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return this->seriesDate;
}

inline const String& DicomSeriesInfo::getSeriesTime()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return this->seriesTime;
}

inline const String& DicomSeriesInfo::getPerformingPhysiciansName()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return this->performingPhysiciansName;
}

inline const String& DicomSeriesInfo::getPerformingPhysicianIdentificationSequence()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return this->performingPhysicianIdentificationSequence;
}

inline const String& DicomSeriesInfo::getOperatorsName()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return this->operatorsName;
}

inline const String& DicomSeriesInfo::getOperatorIdentificationSequence()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return this->operatorIdentificationSequence;
}

inline const String& DicomSeriesInfo::getAcquisitionTime()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return this->acquisitionTime;
}

} // namespace Images
} // namespace TNL
