/***************************************************************************
                          tnlStatistics.cpp  -  description
                             -------------------
    begin                : Feb 10, 2011
    copyright            : (C) 2011 by Tomas Oberhuber
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

#include <core/tnlStatistics.h>

tnlStatistics defaultTnlStatistics;

tnlStatistics :: tnlStatistics()
 : transferedBytes( 0 )
{

}

void tnlStatistics :: reset()
{
   transferedBytes = 0;
}

long int tnlStatistics :: getTransferedBytes() const
{
   return transferedBytes;
}

void tnlStatistics :: addTransferedBytes( const long int transfered )
{
   transferedBytes += transfered;
}
