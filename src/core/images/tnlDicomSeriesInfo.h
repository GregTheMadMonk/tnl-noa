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

#include <core/tnlString.h>
#include <tnlConfig.h>

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

       inline const tnlString& getModality();
 
       inline const tnlString& getStudyInstanceUID();
 
       inline const tnlString& getSeriesInstanceUID();
 
       inline const tnlString& getSeriesDescription();
 
       inline const tnlString& getSeriesNumber();
 
       inline const tnlString& getSeriesDate();
 
       inline const tnlString& getSeriesTime();
 
       inline const tnlString& getPerformingPhysiciansName();
 
       inline const tnlString& getPerformingPhysicianIdentificationSequence();
 
       inline const tnlString& getOperatorsName();
 
       inline const tnlString& getOperatorIdentificationSequence();
 
       inline const tnlString& getAcquisitionTime();
 
   private:
 
       tnlDicomHeader &dicomHeader;
 
       bool retrieveInfo();
 
       bool isObjectRetrieved;

       tnlString modality;

       tnlString studyInstanceUID;

       tnlString seriesInstanceUID;

       tnlString seriesNumber;

       tnlString seriesDescription;

       tnlString seriesDate;

       tnlString seriesTime;

       tnlString performingPhysiciansName;

       tnlString performingPhysicianIdentificationSequence;

       tnlString operatorsName;

       tnlString operatorIdentificationSequence;

       tnlString frameTime;

       tnlString faDateTime;

       tnlString faRefTime;

       tnlString AFD;

       tnlString acquisitionTime;
};

} // namespace TNL

#include <core/images/tnlDicomSeriesInfo_impl.h>

