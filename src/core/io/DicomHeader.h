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

#include <dcmtk/dcmdata/dcfilefo.h>
#include <dcmtk/dcmdata/dcdeftag.h>
#include "ImageInfoObj.h"
#include "PatientInfoObj.h"
#include "SeriesInfoObj.h"

/***
 * Class provides acces to the DICOM file header (contains complete
 *   information about DICOM file) and stores the information objects
 *   focused on essential data about image, patient and serie.
 */
class tnlDicomHeader
{
   public:
      
      tnlDicomHeader();
      virtual ~tnlDicomHeader();

      DcmFileFormat &getFileFormat();
      
      ImageInfoObj &getImageInfoObj();
      
      PatientInfoObj &getPatientInfoObj();
      
      SeriesInfoObj &getSeriesInfoObj();

      bool loadFromFile( const char* fileName );

   protected:
      
      ImageInfoObj *imageInfoObj;
      
      PatientInfoObj *patientInfoObj;
      
      SeriesInfoObj *seriesInfoObj;

      DcmFileFormat *fileFormat;
      
      bool isLoaded;
};

#endif // TNLDICOMHEADER_H
