/***************************************************************************
                          tnlConfigEntryList.h  -  description
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

#ifndef TNLCONFIGENTRYLIST_H_
#define TNLCONFIGENTRYLIST_H_

#include <config/tnlConfigEntryBase.h>

template< typename EntryType >
struct tnlConfigEntryList : public tnlConfigEntryBase
{
   EntryType defaultValue;

   tnlList< EntryType > enumValues;

   public:

   tnlConfigEntryList( const char* name,
                       const char* description,
                       bool required )
      : tnlConfigEntryBase( name,
                            description,
                            required )
      {
         hasDefaultValue = false;
      }

   tnlConfigEntryList( const char* name,
                       const char* description,
                       bool required,
                       const EntryType& defaultValue)
      : tnlConfigEntryBase( name,
                            description,
                            required ),
         defaultValue( defaultValue )
      {
         hasDefaultValue = true;

      }

   tnlString getEntryType() const
   {
      return ::getParameterType< tnlList< EntryType > >();
   }

   tnlString getUIEntryType() const
   {
      return ::getUIEntryType< tnlList< EntryType > >();
   }

   tnlString printDefaultValue() const
   {
      return convertToString( defaultValue );
   };

   tnlList< EntryType >& getEnumValues()
   {
      return this->enumValues;
   }

   bool hasEnumValues() const
   {
      if( enumValues.getSize() != 0 )
         return true;
      return false;
   }

   void printEnumValues() const
   {
      cout << "- Can be:           ";
      int i;
      for( i = 0; i < enumValues.getSize() - 1; i++ )
         cout << enumValues[ i ] << ", ";
      cout << enumValues[ i ];
      cout << " ";
   }

   bool checkValue( const tnlList< EntryType >& values ) const
   {
      if( this->enumValues.getSize() != 0 )
      {
         for( int j = 0; j < values.getSize(); j++ )
         {
            const EntryType& value = values[ j ];
            bool found( false );
            for( int i = 0; i < this->enumValues.getSize(); i++ )
               if( value == this->enumValues[ i ] )
               {
                  found = true;
                  break;
               }
            if( ! found )
            {
               cerr << "The value " << value << " is not allowed for the config entry " << this->name << "." << endl;
               this->printEnumValues();
               return false;
            }
         }
      }
      return true;
   };
};


#endif /* TNLCONFIGENTRYLIST_H_ */
