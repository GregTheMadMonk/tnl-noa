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
#include <TNL/Containers/List.h>
#include <TNL/param-types.h>
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

   /**
    * \brief Basic constructor.
    */
   ConfigDescription();

   /**
    * \brief Adds new entry to the configuration description.
    *
    * \tparam EntryType Type of the entry.
    * \param name Name of the entry.
    * \param description More specific information about the entry.
    */
   template< typename EntryType >
   void addEntry( const String& name,
                  const String& description )
   {
      currentEntry = new ConfigEntry< EntryType >( name, description, false );
      entries.Append( currentEntry );
   }

   /**
    * \brief Adds new entry to the configuration description, that requires set value.
    *
    * \tparam EntryType Type of the entry.
    * \param name Name of the entry.
    * \param description More specific information about the entry.
    */
   template< typename EntryType >
   void addRequiredEntry( const String& name,
                          const String& description )
   {
      currentEntry = new ConfigEntry< EntryType >( name, description, true );
      entries.Append( currentEntry );
   }

   /**
    * \brief Adds new entry to the configuration description.
    *
    * \tparam EntryType Type of the entry.
    * \param name Name of the entry.
    * \param description More specific information about the entry.
    * \param defaultValue Default value of the entry.
    */
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

   /**
    * \brief Adds new list to the configuration description.
    *
    * \tparam EntryType Type of the list.
    * \param name Name of the list.
    * \param description More specific information about the list.
    */
   template< typename EntryType >
   void addList( const String& name,
                 const String& description )
   {
      currentEntry = new ConfigEntryList< EntryType >( name, description, false );
      entries.Append( currentEntry );
   }

   /**
    * \brief Adds new list to the configuration description, that requires specific value.
    *
    * \tparam EntryType Type of the list.
    * \param name Name of the list.
    * \param description More specific information about the list.
    */
   template< typename EntryType >
   void addRequiredList( const String& name,
                         const String& description )
   {
      currentEntry = new ConfigEntryList< EntryType >( name, description, true );
      entries.Append( currentEntry );
   }

   /**
    * \brief Adds new list to the configuration description.
    *
    * \tparam EntryType Type of the list.
    * \param name Name of the list.
    * \param description More specific information about the list.
    * \param defaultValue Default value of the list.
    */
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

   /**
    * \brief Adds new entry enumeration of type \e EntryType.
    *
    * Adds new option of setting an entry value.
    * \tparam EntryType Type of the entry enumeration.
    * \param entryEnum Value of the entry enumeration.
    */
   template< typename EntryType >
   void addEntryEnum( const EntryType& entryEnum )
   {
      TNL_ASSERT( this->currentEntry,);
      ( ( ConfigEntry< EntryType >* ) currentEntry )->getEnumValues().Append( entryEnum );
   }

   /**
    * \brief Adds new entry enumeration of type \e char.
    *
    * Adds new option of setting an entry value.
    * \param entryEnum Value of the entry enumeration.
    */
   void addEntryEnum( const char* entryEnum )
   {
      TNL_ASSERT( this->currentEntry,);
      ( ( ConfigEntry< String >* ) currentEntry )->getEnumValues().Append( String( entryEnum ) );
   }

   /**
    * \brief Adds delimeter/section to the configuration description.
    *
    * \param delimeter String that defines how the delimeter looks like.
    */
   void addDelimiter( const String& delimiter )
   {
      entries.Append( new ConfigDelimiter( delimiter ) );
      currentEntry = 0;
   }

   /**
    * \brief Gets entry out of the configuration description.
    *
    * \param name Name of the entry.
    */
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
      std::cerr << "Asking for the default value of unknown parameter." << std::endl;
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
      std::cerr << "Asking for the default value of unknown parameter." << std::endl;
      return NULL;
   }

   /**
    * \brief Fills in the parameters from the \e parameter_container.
    *
    * Parameters which were not defined in the command line by user but have their default value are added to the congiguration description.
    * If there is missing entry with defined default value in the Config::ParameterContainer it is going to be added.
    * \param parameter_container Name of the ParameterContainer object.
    */
   void addMissingEntries( Config::ParameterContainer& parameter_container ) const;

   //! Check for all entries with the flag 'required'.
   /*! Returns false if any parameter is missing.
    */
   bool checkMissingEntries( Config::ParameterContainer& parameter_container,
                             bool printUsage,
                             const char* programName ) const;

   /**
    * \brief Prints configuration description with the \e program_name at the top.
    *
    * \param program_name Name of the program
    */
   void printUsage( const char* program_name ) const;

   //bool parseConfigDescription( const char* file_name );

   /**
    * \brief Basic destructor.
    */
   ~ConfigDescription();

   protected:

   Containers::List< ConfigEntryBase* > entries;

   ConfigEntryBase* currentEntry;

};

} //namespace Config
} //namespace TNL

