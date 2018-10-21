/***************************************************************************
                          ConfigEntry.h  -  description
                             -------------------
    begin                : Jul 5, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <vector>

#include <TNL/Config/ConfigEntryBase.h>

namespace TNL {
namespace Config {

template< typename EntryType >
struct ConfigEntry : public ConfigEntryBase
{
   EntryType defaultValue;

   std::vector< EntryType > enumValues;

   public:

   ConfigEntry( const String& name,
                const String& description,
                bool required )
   : ConfigEntryBase( name,
                      description,
                      required )
   {
      hasDefaultValue = false;
   }

   ConfigEntry( const String& name,
                const String& description,
                bool required,
                const EntryType& defaultValue)
   : ConfigEntryBase( name,
                      description,
                      required ),
     defaultValue( defaultValue )
   {
      hasDefaultValue = true;
   }

   String getEntryType() const
   {
      return TNL::getType< EntryType >();
   }

   String getUIEntryType() const
   {
      return TNL::Config::getUIEntryType< EntryType >();
   }

   String printDefaultValue() const
   {
      return convertToString( defaultValue );
   };

   std::vector< EntryType >& getEnumValues()
   {
      return this->enumValues;
   }

   bool hasEnumValues() const
   {
      if( enumValues.size() > 0 )
         return true;
      return false;
   }

   void printEnumValues() const
   {
      std::cout << "- Can be:           ";
      int i;
      for( i = 0; i < (int) enumValues.size() - 1; i++ )
         std::cout << enumValues[ i ] << ", ";
      std::cout << enumValues[ i ];
      std::cout << " ";
   }

   bool checkValue( const EntryType& value ) const
   {
      if( this->enumValues.size() > 0 )
      {
         bool found( false );
         for( int i = 0; i < (int) this->enumValues.size(); i++ )
            if( value == this->enumValues[ i ] )
            {
               found = true;
               break;
            }
         if( ! found )
         {
            std::cerr << "The value " << value << " is not allowed for the config entry " << this->name << "." << std::endl;
            this->printEnumValues();
            std::cerr << std::endl;
            return false;
         }
      }
      return true;
   }
};

} // namespace Config
} // namespace TNL
