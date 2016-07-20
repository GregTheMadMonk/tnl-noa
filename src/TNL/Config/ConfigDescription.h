/***************************************************************************
                          Config::ConfigDescription.h  -  description
                             -------------------
    begin                : 2007/06/09
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/String.h>
#include <TNL/List.h>
#include <TNL/core/param-types.h>
#include <TNL/Config/ConfigEntryType.h>
#include <TNL/Config/ConfigEntry.h>
#include <TNL/Config/ConfigEntryList.h>
#include <TNL/Config/ConfigDelimiter.h>

namespace TNL {
namespace Config {

class ParameterContainer;

class ConfigDescription
{
   public:

   ConfigDescription();

   template< typename EntryType >
   void addEntry( const String& name,
                  const String& description )
   {
      currentEntry = new ConfigEntry< EntryType >( name, description, false );
      entries.Append( currentEntry );
   }

   template< typename EntryType >
   void addRequiredEntry( const String& name,
                          const String& description )
   {
      currentEntry = new ConfigEntry< EntryType >( name, description, true );
      entries.Append( currentEntry );
   }
 
   template< typename EntryType >
   void addEntry( const String& name,
                  const String& description,
                  const EntryType& defaultValue )
   {
      currentEntry = new ConfigEntry< EntryType >( name,
                                                      description,
                                                      false,
                                                      defaultValue );
      entries. Append( currentEntry );
   }

   template< typename EntryType >
   void addList( const String& name,
                 const String& description )
   {
      currentEntry = new ConfigEntryList< EntryType >( name, description, false );
      entries.Append( currentEntry );
   }

   template< typename EntryType >
   void addRequiredList( const String& name,
                         const String& description )
   {
      currentEntry = new ConfigEntryList< EntryType >( name, description, true );
      entries.Append( currentEntry );
   }

   template< typename EntryType >
   void addList( const String& name,
                 const String& description,
                 const EntryType& defaultValue )
   {
      currentEntry = new ConfigEntryList< EntryType >( name,
                                                          description,
                                                          false,
                                                          defaultValue );
      entries. Append( currentEntry );
   }

   template< typename EntryType >
   void addEntryEnum( const EntryType& entryEnum )
   {
      Assert( this->currentEntry,);
      ( ( ConfigEntry< EntryType >* ) currentEntry )->getEnumValues().Append( entryEnum );
   }

   void addEntryEnum( const char* entryEnum )
   {
      Assert( this->currentEntry,);
      ( ( ConfigEntry< String >* ) currentEntry )->getEnumValues().Append( String( entryEnum ) );
   }

   void addDelimiter( const String& delimiter )
   {
      entries.Append( new ConfigDelimiter( delimiter ) );
      currentEntry = 0;
   }

   const ConfigEntryBase* getEntry( const String& name ) const
   {
      for( int i = 0; i < entries.getSize(); i++ )
         if( entries[ i ]->name == name )
            return entries[ i ];
      return NULL;
   }

 
   //! Returns empty string if given entry does not exist
   //const String getEntryType( const char* name ) const;

   //! Returns zero pointer if there is no default value
   template< class T > const T* getDefaultValue( const String& name ) const
   {
      int i;
      const int entries_num = entries. getSize();
      for( i = 0; i < entries_num; i ++ )
         if( entries[ i ] -> name == name )
         {
            if( entries[ i ] -> hasDefaultValue )
               return ( ( ConfigEntry< T > * ) entries[ i ] ) -> default_value;
            else return NULL;
         }
      std::cerr << "Asking for the default value of uknown parameter." << std::endl;
      return NULL;
   }
 
   //! Returns zero pointer if there is no default value
   template< class T > T* getDefaultValue( const String& name )
   {
      int i;
      const int entries_num = entries. getSize();
      for( i = 0; i < entries_num; i ++ )
         if( entries[ i ] -> name == name )
         {
            if( entries[ i ] -> hasDefaultValue )
               return ( ( ConfigEntry< T > * ) entries[ i ] ) -> default_value;
            else return NULL;
         }
      std::cerr << "Asking for the default value of uknown parameter." << std::endl;
      return NULL;
   }

   //! If there is missing entry with defined default value in the Config::ParameterContainer it is going to be added
   void addMissingEntries( Config::ParameterContainer& parameter_container ) const;

   //! Check for all entries with the flag 'required'.
   /*! Returns false if any parameter is missing.
    */
   bool checkMissingEntries( Config::ParameterContainer& parameter_container,
                             bool printUsage,
                             const char* programName ) const;

   void printUsage( const char* program_name ) const;

   //bool parseConfigDescription( const char* file_name );

   ~ConfigDescription();

   protected:

   List< ConfigEntryBase* > entries;

   ConfigEntryBase* currentEntry;

};

} //namespace Config
} //namespace TNL

