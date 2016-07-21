/***************************************************************************
                          tnlDicomSeries.h  -  description
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
#include <dcmtk/dcmdata/dcfilefo.h>
#include <dcmtk/dcmdata/dcdeftag.h>
#define HAVE_STD_STRING
#include <dcmtk/ofstd/ofstring.h>
#endif

namespace TNL {

class tnlDicomHeader;

/***
 * SeriesInfoObj class stores selected informations about DICOM series.
 * (accesses information via tnlDicomHeader class)
 */
class tnlDicomSeriesInfo
{
   public:
 
       inline tnlDicomSeriesInfo( tnlDicomHeader &dicomHeader );
 
       inline virtual ~tnlDicomSeriesInfo();

       inline const String& getModality();
 
       inline const String& getStudyInstanceUID();
 
       inline const String& getSeriesInstanceUID();
 
       inline const String& getSeriesDescription();
 
       inline const String& getSeriesNumber();
 
       inline const String& getSeriesDate();
 
       inline const String& getSeriesTime();
 
       inline const String& getPerformingPhysiciansName();
 
       inline const String& getPerformingPhysicianIdentificationSequence();
 
       inline const String& getOperatorsName();
 
       inline const String& getOperatorIdentificationSequence();
 
       inline const String& getAcquisitionTime();
 
   private:
 
       tnlDicomHeader &dicomHeader;
 
       bool retrieveInfo();
 
       bool isObjectRetrieved;

       String modality;

       String studyInstanceUID;

       String seriesInstanceUID;

       String seriesNumber;

       String seriesDescription;

       String seriesDate;

       String seriesTime;

       String performingPhysiciansName;

       String performingPhysicianIdentificationSequence;

       String operatorsName;

       String operatorIdentificationSequence;

       String frameTime;

       String faDateTime;

       String faRefTime;

       String AFD;

       String acquisitionTime;
};

} // namespace TNL

#include <TNL/core/images/tnlDicomSeriesInfo_impl.h>

