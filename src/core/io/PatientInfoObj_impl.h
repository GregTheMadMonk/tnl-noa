#include "PatientInfoObj.h"
#include "DicomHeader.h"

#ifdef HAVE_DCMTK_H
#define HAVE_CONFIG_H
#define HAVE_STD_STRING
#include <dcmtk/ofstd/ofstring.h>
#endif

inline PatientInfoObj::PatientInfoObj( tnlDicomHeader &dicomHeader )
: dicomHeader( dicomHeader )
{
    isObjectRetrieved = false;
}

inline PatientInfoObj::~PatientInfoObj()
{
}

inline bool PatientInfoObj::retrieveInfo()
{
   OFString str;
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString(DCM_PatientName, str );
   this->name.setString( str.data() );
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString(DCM_PatientSex, str );
   this->sex.setString( str.data() );
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString(DCM_PatientID, str );
   this->ID.setString( str.data() );
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString(DCM_PatientWeight, str );
   this->weight.setString( str.data() );
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString(DCM_PatientPosition, str );
   this->patientPosition.setString( str.data() ); 
   dicomHeader.getFileFormat().getDataset()->findAndGetOFString(DCM_PatientOrientation, str );
   this->patientOrientation.setString( str.data() ); 

   isObjectRetrieved = true;
   return 0;
}

inline const tnlString& PatientInfoObj::getName()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return name;
}

inline const tnlString& PatientInfoObj::getSex()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return sex;
}

inline const tnlString& PatientInfoObj::getID()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return ID;
}

inline const tnlString& PatientInfoObj::getWeight()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return weight;
}

inline const tnlString& PatientInfoObj::getPosition()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return patientPosition;
}

inline const tnlString& PatientInfoObj::getOrientation()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return patientOrientation;
}
