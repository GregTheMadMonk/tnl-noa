/***************************************************************************
                          tnlStatistics.h  -  description
                             -------------------
    begin                : Feb 10, 2011
    copyright            : (C) 2011 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {

class tnlStatistics
{
   protected:

   long int transferedBytes;

   public:

   //! Default constructor
   tnlStatistics();

   void reset();

   long int getTransferedBytes() const;

   void addTransferedBytes( const long int );
};

extern tnlStatistics defaultTnlStatistics;

} // namespace TNL
