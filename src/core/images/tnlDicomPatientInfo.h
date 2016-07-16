/***************************************************************************
                          tnlDicomPatientInfo.h  -  description
                             -------------------
    begin                : Jul 19, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
 
     Tomas Oberhuber     tomas.oberhuber@fjfi.cvut.cz
     Jiri Kafka          kafka9@seznam.cz
     Pavel Neskudla
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */


#ifndef TNLDICOMPATIENTINFO_H
#define TNLDICOMPATIENTINFO_H

class tnlDicomHeader;

#include <core/tnlString.h>
#include <tnlConfig.h>

#ifdef HAVE_DCMTK_H
#define HAVE_CONFIG_H
#define HAVE_STD_STRING
#include <dcmtk/dcmdata/dcfilefo.h>
#include <dcmtk/dcmdata/dcdeftag.h>
#include <dcmtk/ofstd/ofstring.h>
#endif

/***
 * PatientInfoObj class stores selected informations about patient.
 * (accesses information via tnlDicomHeader class)
 */
class tnlDicomPatientInfo
{
   public:
 
      inline tnlDicomPatientInfo(tnlDicomHeader &atnlDicomHeader);
 
      inline virtual ~tnlDicomPatientInfo();

      inline const tnlString& getName();
 
      inline const tnlString& getSex();
 
      inline const tnlString& getID();
 
      inline const tnlString& getWeight();
 
      inline const tnlString& getPosition();
 
      inline const tnlString& getOrientation();

   private:

       tnlDicomHeader &dicomHeader;
       bool retrieveInfo();
       bool isObjectRetrieved;

       tnlString name;

       tnlString sex;

       tnlString ID;

       tnlString weight;

       tnlString patientPosition;

       tnlString patientOrientation;
};

#include <core/images/tnlDicomPatientInfo_impl.h>

#endif // TNLDICOMPATIENTINFO_H
