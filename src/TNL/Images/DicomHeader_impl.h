/***************************************************************************
                          DicomHeader_impl.h  -  description
                             -------------------
    begin                : Jul 19, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
 
     Tomas Oberhuber     tomas.oberhuber@fjfi.cvut.cz
     Jiri Kafka          kafka9@seznam.cz
     Pavel Neskudla
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Images//DicomHeader.h>
#include <TNL/Images//DicomSeriesInfo.h>
#include <TNL/Images//DicomPatientInfo.h>
#include <TNL/Images//DicomImageInfo.h>

namespace TNL {
namespace Images {

inline DicomHeader::DicomHeader()
{
#ifdef HAVE_DCMTK_H
    fileFormat = new DcmFileFormat();
#endif
    isLoaded = false;
    imageInfoObj = new DicomImageInfo(*this);
    patientInfoObj = new DicomPatientInfo(*this);
    seriesInfoObj = new DicomSeriesInfo(*this);
}

inline DicomHeader::~DicomHeader()
{
    delete imageInfoObj;
    delete patientInfoObj;
    delete seriesInfoObj;
#ifdef HAVE_DCMTK_H
    delete fileFormat;
#endif
}

inline bool DicomHeader::loadFromFile( const String& fileName )
{
#ifdef HAVE_DCMTK_H
    OFCondition status = fileFormat->loadFile( fileName.getString() );
    if(status.good())
    {
        isLoaded = true;
        return true;
    }
#endif
    isLoaded = false;
    return false;
}

#ifdef HAVE_DCMTK_H
inline DcmFileFormat &DicomHeader::getFileFormat()
{
    return *fileFormat;
}
#endif

inline DicomImageInfo &DicomHeader::getImageInfo()
{
    return *imageInfoObj;
}

inline DicomPatientInfo &DicomHeader::getPatientInfo()
{
    return *patientInfoObj;
}

inline DicomSeriesInfo &DicomHeader::getSeriesInfo()
{
    return *seriesInfoObj;
}

} // namespace Images
} // namespace TNL

