/***************************************************************************
                          tnlConfigDelimiter.h  -  description
                             -------------------
    begin                : Jul 5, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
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

#ifndef TNLCONFIGDELIMITER_H_
#define TNLCONFIGDELIMITER_H_

struct tnlConfigDelimiter : public tnlConfigEntryBase
{
   tnlConfigDelimiter( const tnlString& delimiter )
   : tnlConfigEntryBase( "", delimiter, false )
   {
   };

   bool isDelimiter() const { return true; };

   tnlString getEntryType() const { return ""; };

   tnlString getUIEntryType() const { return ""; };
};

#endif /* TNLCONFIGDELIMITER_H_ */
