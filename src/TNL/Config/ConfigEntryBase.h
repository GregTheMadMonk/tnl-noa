/***************************************************************************
                          ConfigEntryBase.h  -  description
                             -------------------
    begin                : Jul 5, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Config {

struct ConfigEntryBase
{
   String name;

   String description;

   bool required;

   bool hasDefaultValue;

   ConfigEntryBase( const String& name,
                    const String& description,
                    bool required )
      : name( name ),
        description( description ),
        required( required ),
        hasDefaultValue( false )
   {}

   virtual String getEntryType() const = 0;

   virtual String getUIEntryType() const = 0;

   virtual bool isDelimiter() const { return false; }

   virtual String printDefaultValue() const { return ""; }

   virtual bool hasEnumValues() const { return false; }

   virtual void printEnumValues() const {}

   virtual ~ConfigEntryBase() = default;
};

} // namespace Config
} // namespace TNL
