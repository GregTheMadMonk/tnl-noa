#ifndef PATIENTINFOOBJ_H
#define PATIENTINFOOBJ_H

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
class PatientInfoObj
{
   public:
      
      inline PatientInfoObj(tnlDicomHeader &atnlDicomHeader);
       
      inline virtual ~PatientInfoObj();

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

#include <core/io/PatientInfoObj_impl.h>

#endif // PATIENTINFOOBJ_H
