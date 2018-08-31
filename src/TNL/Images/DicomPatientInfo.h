/***************************************************************************
                          DicomPatientInfo.h  -  description
                             -------------------
    begin                : Jul 19, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
 
     Tomas Oberhuber     tomas.oberhuber@fjfi.cvut.cz
     Jiri Kafka          kafka9@seznam.cz
     Pavel Neskudla
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/String.h>
#include <TNL/tnlConfig.h>

#ifdef HAVE_DCMTK_H
#define HAVE_CONFIG_H
#define HAVE_STD_STRING
#include <dcmtk/dcmdata/dcfilefo.h>
#include <dcmtk/dcmdata/dcdeftag.h>
#include <dcmtk/ofstd/ofstring.h>
#endif

namespace TNL {
namespace Images {   

class DicomHeader;

/***
 * PatientInfoObj class stores selected informations about patient.
 * (accesses information via DicomHeader class)
 */
class DicomPatientInfo
{
   public:
 
      inline DicomPatientInfo(DicomHeader &aDicomHeader);
 
      inline virtual ~DicomPatientInfo();

      inline const String& getName();
 
      inline const String& getSex();
 
      inline const String& getID();
 
      inline const String& getWeight();
 
      inline const String& getPosition();
 
      inline const String& getOrientation();

   private:

       DicomHeader &dicomHeader;
       bool retrieveInfo();
       bool isObjectRetrieved;

       String name;

       String sex;

       String ID;

       String weight;

       String patientPosition;

       String patientOrientation;
};

} // namespace Images
} // namespace TNL

#include <TNL/Images//DicomPatientInfo_impl.h>

