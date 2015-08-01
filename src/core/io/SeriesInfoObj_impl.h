#include "SeriesInfoObj.h"
#include <core/io/DicomHeader.h>
#include <stdio.h>

inline SeriesInfoObj::SeriesInfoObj( tnlDicomHeader &dicomHeader)
: dicomHeader( dicomHeader )
{
    isObjectRetrieved = false;
}

inline SeriesInfoObj::~SeriesInfoObj()
{
}

inline bool SeriesInfoObj::retrieveInfo()
{
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
    return 0;
}

inline const tnlString& SeriesInfoObj::getModality()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return this->modality;
}

inline const tnlString& SeriesInfoObj::getStudyInstanceUID()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return this->studyInstanceUID;
}

inline const tnlString& SeriesInfoObj::getSeriesInstanceUID()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return this->seriesInstanceUID;
}

inline const tnlString& SeriesInfoObj::getSeriesNumber()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return this->seriesNumber;
}

inline const tnlString& SeriesInfoObj::getSeriesDescription()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return this->seriesDescription;
}

inline const tnlString& SeriesInfoObj::getSeriesDate()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return this->seriesDate;
}

inline const tnlString& SeriesInfoObj::getSeriesTime()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return this->seriesTime;
}

inline const tnlString& SeriesInfoObj::getPerformingPhysiciansName()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return this->performingPhysiciansName;
}

inline const tnlString& SeriesInfoObj::getPerformingPhysicianIdentificationSequence()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return this->performingPhysicianIdentificationSequence;
}

inline const tnlString& SeriesInfoObj::getOperatorsName()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return this->operatorsName;
}

inline const tnlString& SeriesInfoObj::getOperatorIdentificationSequence()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return this->operatorIdentificationSequence;
}

inline const tnlString& SeriesInfoObj::getAcquisitionTime()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return this->acquisitionTime;
}
