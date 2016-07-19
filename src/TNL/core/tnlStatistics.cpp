/***************************************************************************
                          tnlStatistics.cpp  -  description
                             -------------------
    begin                : Feb 10, 2011
    copyright            : (C) 2011 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/core/tnlStatistics.h>

namespace TNL {

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

} // namespace TNL
