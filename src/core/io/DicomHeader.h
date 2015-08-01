/***************************************************************************
                          tnlDicomHeader.h  -  description
                             -------------------
    begin                : Jul 19, 2015
    copyright            : (C) 2015 by Jiri Kafka,
                                       Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLDICOMHEADER_H
#define TNLDICOMHEADER_H

#include <tnlConfig.h>

#ifdef HAVE_DCMTK_H
#define HAVE_CONFIG_H
#include <dcmtk/dcmdata/dcfilefo.h>
#include <dcmtk/dcmdata/dcdeftag.h>
#endif

class SeriesInfoObj;
class PatientInfoObj;
class ImageInfoObj;

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
      
      inline ImageInfoObj &getImageInfoObj();
      
      inline PatientInfoObj &getPatientInfoObj();
      
      inline SeriesInfoObj &getSeriesInfoObj();

      inline bool loadFromFile( const char* fileName );

   protected:
      
      ImageInfoObj *imageInfoObj;
      
      PatientInfoObj *patientInfoObj;
      
      SeriesInfoObj *seriesInfoObj;

#ifdef HAVE_DCMTK_H      
      DcmFileFormat *fileFormat;
#endif
      
      bool isLoaded;
};

#include <core/io/DicomHeader_impl.h>

#endif // TNLDICOMHEADER_H
