/***************************************************************************
                          DicomHeader.h  -  description
                             -------------------
    begin                : Jul 19, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
 
     Tomas Oberhuber     tomas.oberhuber@fjfi.cvut.cz
     Jiri Kafka          kafka9@seznam.cz
     Pavel Neskudla
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

#ifdef HAVE_DCMTK_H
#define HAVE_CONFIG_H
#include <dcmtk/dcmdata/dcfilefo.h>
#include <dcmtk/dcmdata/dcdeftag.h>
#endif

namespace TNL {
namespace Images {

class DicomSeriesInfo;
class DicomPatientInfo;
class DicomImageInfo;

/***
 * Class provides acces to the DICOM file header (contains complete
 *   information about DICOM file) and stores the information objects
 *   focused on essential data about image, patient and serie.
 */
class DicomHeader
{
   public:
 
      inline DicomHeader();
 
      inline virtual ~DicomHeader();

#ifdef HAVE_DCMTK_H
      inline DcmFileFormat &getFileFormat();
#endif
 
      inline DicomImageInfo &getImageInfo();
 
      inline DicomPatientInfo &getPatientInfo();
 
      inline DicomSeriesInfo &getSeriesInfo();

      inline bool loadFromFile( const String& fileName );

   protected:
 
      DicomImageInfo *imageInfoObj;
 
      DicomPatientInfo *patientInfoObj;
 
      DicomSeriesInfo *seriesInfoObj;

#ifdef HAVE_DCMTK_H
      DcmFileFormat *fileFormat;
#endif
 
      bool isLoaded;
};

} // namespace Images
} // namespace TNL

#include <TNL/Images//DicomHeader_impl.h>

