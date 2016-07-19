/***************************************************************************
                          tnlConfigEntry.h  -  description
                             -------------------
    begin                : Jul 5, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

#include <TNL/config/tnlConfigEntryBase.h>

namespace TNL {

template< typename EntryType >
struct tnlConfigEntry : public tnlConfigEntryBase
{
   EntryType defaultValue;

   tnlList< EntryType > enumValues;

   public:

   tnlConfigEntry( const tnlString& name,
                   const tnlString& description,
                   bool required )
      : tnlConfigEntryBase( name,
                            description,
                            required )
      {
         hasDefaultValue = false;
      }

   tnlConfigEntry( const tnlString& name,
                   const tnlString& description,
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
      return TNL::getType< EntryType >();
   }

   tnlString getUIEntryType() const
   {
      return TNL::getUIEntryType< EntryType >();
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
     std::cout << "- Can be:           ";
      int i;
      for( i = 0; i < enumValues.getSize() - 1; i++ )
        std::cout << enumValues[ i ] << ", ";
     std::cout << enumValues[ i ];
     std::cout << " ";
   }

   bool checkValue( const EntryType& value ) const
   {
      if( this->enumValues.getSize() != 0 )
      {
         bool found( false );
         for( int i = 0; i < this->enumValues.getSize(); i++ )
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
   };
};

} // namespace TNL
