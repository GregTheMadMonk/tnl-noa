/***************************************************************************
                          tnlStatistics.h  -  description
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

#ifndef TNLSTATISTICS_H_
#define TNLSTATISTICS_H_


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

#endif /* TNLSTATISTICS_H_ */
