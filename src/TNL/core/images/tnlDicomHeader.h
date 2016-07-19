/***************************************************************************
                          tnlDicomHeader.h  -  description
                             -------------------
    begin                : Jul 19, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
 
     Tomas Oberhuber     tomas.oberhuber@fjfi.cvut.cz
     Jiri Kafka          kafka9@seznam.cz
     Pavel Neskudla
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

#include <TNL/tnlConfig.h>

#ifdef HAVE_DCMTK_H
#define HAVE_CONFIG_H
#include <dcmtk/dcmdata/dcfilefo.h>
#include <dcmtk/dcmdata/dcdeftag.h>
#endif

namespace TNL {

class tnlDicomSeriesInfo;
class tnlDicomPatientInfo;
class tnlDicomImageInfo;

/***
 * Class provides acces to the DICOM file header (contains complete
 *   information about DICOM file) and stores the information objects
 *   focused on essential data about image, patient and serie.
 */
class tnlDicomHeader
{
   public:
 
      inline tnlDicomHeader();
 
      inline virtual ~tnlDicomHeader();

#ifdef HAVE_DCMTK_H
      inline DcmFileFormat &getFileFormat();
#endif
 
      inline tnlDicomImageInfo &getImageInfo();
 
      inline tnlDicomPatientInfo &getPatientInfo();
 
      inline tnlDicomSeriesInfo &getSeriesInfo();

      inline bool loadFromFile( const tnlString& fileName );

   protected:
 
      tnlDicomImageInfo *imageInfoObj;
 
      tnlDicomPatientInfo *patientInfoObj;
 
      tnlDicomSeriesInfo *seriesInfoObj;

#ifdef HAVE_DCMTK_H
      DcmFileFormat *fileFormat;
#endif
 
      bool isLoaded;
};

} // namespace TNL

#include <TNL/core/images/tnlDicomHeader_impl.h>

