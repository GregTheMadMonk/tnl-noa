/***************************************************************************
                          tnlDicomPatientInfo_impl.h  -  description
                             -------------------
    begin                : Jul 19, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
 
     Tomas Oberhuber     tomas.oberhuber@fjfi.cvut.cz
     Jiri Kafka          kafka9@seznam.cz
     Pavel Neskudla
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include "tnlDicomPatientInfo.h"
#include "tnlDicomHeader.h"

#ifdef HAVE_DCMTK_H
#define HAVE_CONFIG_H
#define HAVE_STD_STRING
#include <dcmtk/ofstd/ofstring.h>
#endif

namespace TNL {

inline tnlDicomPatientInfo::tnlDicomPatientInfo( tnlDicomHeader &dicomHeader )
: dicomHeader( dicomHeader )
{
    isObjectRetrieved = false;
}

inline tnlDicomPatientInfo::~tnlDicomPatientInfo()
{
}

inline bool tnlDicomPatientInfo::retrieveInfo()
{
#ifdef HAVE_DCMTK_H
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
   return true;
#else
   std::cerr << "DICOM format is not supported in this build of TNL." << std::endl;
   return false;
#endif
}

inline const String& tnlDicomPatientInfo::getName()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return name;
}

inline const String& tnlDicomPatientInfo::getSex()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return sex;
}

inline const String& tnlDicomPatientInfo::getID()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return ID;
}

inline const String& tnlDicomPatientInfo::getWeight()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return weight;
}

inline const String& tnlDicomPatientInfo::getPosition()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return patientPosition;
}

inline const String& tnlDicomPatientInfo::getOrientation()
{
    if(!isObjectRetrieved)
        retrieveInfo();
    return patientOrientation;
}

} // namespace TNL
