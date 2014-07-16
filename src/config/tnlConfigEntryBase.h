/***************************************************************************
                          tnlConfigEntryBase.h  -  description
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

#ifndef TNLCONFIGENTRYBASE_H_
#define TNLCONFIGENTRYBASE_H_

struct tnlConfigEntryBase
{
   tnlString name;

   tnlString description;

   bool required;

   bool hasDefaultValue;

   tnlConfigEntryBase( const char* name,
                       const char* description,
                       bool required )
      : name( name ),
        description( description ),
        required( required ),
        hasDefaultValue( false ){}

   virtual tnlString getEntryType() const = 0;

   virtual tnlString getUIEntryType() const = 0;

   virtual bool isDelimiter() const { return false; };

   virtual tnlString printDefaultValue() const { return "";};

   virtual bool hasEnumValues() const { return false; };

   virtual void printEnumValues() const{};
};

#endif /* TNLCONFIGENTRYBASE_H_ */
