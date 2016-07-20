/***************************************************************************
                          tnlDicomSeriesInfo_impl.h  -  description
                             -------------------
    begin                : Jul 19, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
 
     Tomas Oberhuber     tomas.oberhuber@fjfi.cvut.cz
     Jiri Kafka          kafka9@seznam.cz
     Pavel Neskudla
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/core/images/tnlDicomSeriesInfo.h>
#include <TNL/core/images/tnlDicomHeader.h>
#include <stdio.h>

namespace TNL {

inline tnlDicomSeriesInfo::tnlDicomSeriesInfo( tnlDicomHeader &dicomHeader)
: dicomHeader( dicomHeader )
{
    isObjectRetrieved = false;
}

inline tnlDicomSeriesInfo::~tnlDicomSeriesInfo()
{
}

inline bool tnlDicomSeriesInfo::retrieveInfo()
{
#ifdef HAVE_DCMTK_H
   OFString str;
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_Modality, str );
   this->modality.setString( str.data() );
 
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_StudyInstanceUID, str );
   this->studyInstanceUID.setString( str.data() );
 
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_SeriesInstanceUID, str );
   this->seriesInstanceUID.setString( str.data() );
 
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_SeriesNumber, str );
   this->seriesNumber.setString( str.data() );
 
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_SeriesDescription, str );
   this->seriesDescription.setString( str.data() );
 
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_SeriesDate, str );
   this->seriesDate.setString( str.data() );
 
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_SeriesTime, str );
   this->seriesTime.setString( str.data() );
 
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_PerformingPhysicianName, str );
   this->performingPhysiciansName.setString( str.data() );
 
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_PerformingPhysicianIdentificationSequence, str );
   this->performingPhysicianIdentificationSequence.setString( str.data() );
 
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_OperatorsName, str );
   this->operatorsName.setString( str.data() );
 
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_OperatorIdentificationSequence, str );
   this->operatorIdentificationSequence.setString( str.data());
 
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_FrameAcquisitionDuration, str );
   this->frameTime.setString( str.data() );
 
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_FrameAcquisitionDateTime, str );
   this->faDateTime.setString( str.data() );
 
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_FrameReferenceTime, str );
   this->faRefTime.setString( str.data() );
 
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_ActualFrameDuration, str );
   this->AFD.setString( str.data() );
 
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString( DCM_AcquisitionTime, str );
   this->acquisitionTime.setString( str.data() );

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

inline const String& tnlDicomSeriesInfo::getModality()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return this->modality;
}

inline const String& tnlDicomSeriesInfo::getStudyInstanceUID()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return this->studyInstanceUID;
}

inline const String& tnlDicomSeriesInfo::getSeriesInstanceUID()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return this->seriesInstanceUID;
}

inline const String& tnlDicomSeriesInfo::getSeriesNumber()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return this->seriesNumber;
}

inline const String& tnlDicomSeriesInfo::getSeriesDescription()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return this->seriesDescription;
}

inline const String& tnlDicomSeriesInfo::getSeriesDate()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return this->seriesDate;
}

inline const String& tnlDicomSeriesInfo::getSeriesTime()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return this->seriesTime;
}

inline const String& tnlDicomSeriesInfo::getPerformingPhysiciansName()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return this->performingPhysiciansName;
}

inline const String& tnlDicomSeriesInfo::getPerformingPhysicianIdentificationSequence()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return this->performingPhysicianIdentificationSequence;
}

inline const String& tnlDicomSeriesInfo::getOperatorsName()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return this->operatorsName;
}

inline const String& tnlDicomSeriesInfo::getOperatorIdentificationSequence()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return this->operatorIdentificationSequence;
}

inline const String& tnlDicomSeriesInfo::getAcquisitionTime()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return this->acquisitionTime;
}

} // namespace TNL
