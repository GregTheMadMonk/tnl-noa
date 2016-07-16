/***************************************************************************
                          tnlConfigEntryBase.h  -  description
                             -------------------
    begin                : Jul 5, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLCONFIGENTRYBASE_H_
#define TNLCONFIGENTRYBASE_H_

struct tnlConfigEntryBase
{
   tnlString name;

   tnlString description;

   bool required;

   bool hasDefaultValue;

   tnlConfigEntryBase( const tnlString& name,
                       const tnlString& description,
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
