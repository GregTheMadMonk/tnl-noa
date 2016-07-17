/***************************************************************************
                          tnlDicomHeader_impl.h  -  description
                             -------------------
    begin                : Jul 19, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
 
     Tomas Oberhuber     tomas.oberhuber@fjfi.cvut.cz
     Jiri Kafka          kafka9@seznam.cz
     Pavel Neskudla
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <core/images/tnlDicomHeader.h>
#include <core/images/tnlDicomSeriesInfo.h>
#include <core/images/tnlDicomPatientInfo.h>
#include <core/images/tnlDicomImageInfo.h>

namespace TNL {

inline tnlDicomHeader::tnlDicomHeader()
{
#ifdef HAVE_DCMTK_H
    fileFormat = new DcmFileFormat();
#endif
    isLoaded = false;
    imageInfoObj = new tnlDicomImageInfo(*this);
    patientInfoObj = new tnlDicomPatientInfo(*this);
    seriesInfoObj = new tnlDicomSeriesInfo(*this);
}

inline tnlDicomHeader::~tnlDicomHeader()
{
    delete imageInfoObj;
    delete patientInfoObj;
    delete seriesInfoObj;
#ifdef HAVE_DCMTK_H
    delete fileFormat;
#endif
}

inline bool tnlDicomHeader::loadFromFile( const tnlString& fileName )
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
inline DcmFileFormat &tnlDicomHeader::getFileFormat()
{
    return *fileFormat;
}
#endif

inline tnlDicomImageInfo &tnlDicomHeader::getImageInfo()
{
    return *imageInfoObj;
}

inline tnlDicomPatientInfo &tnlDicomHeader::getPatientInfo()
{
    return *patientInfoObj;
}

inline tnlDicomSeriesInfo &tnlDicomHeader::getSeriesInfo()
{
    return *seriesInfoObj;
}

} // namespace TNL

