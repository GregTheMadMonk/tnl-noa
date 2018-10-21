/***************************************************************************
                          Config::ConfigDescription.h  -  description
                             -------------------
    begin                : 2007/06/09
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <vector>
#include <memory>

// std::make_unique does not exist until C++14
// https://stackoverflow.com/a/9657991
#if __cplusplus < 201402L
namespace std {
   template<typename T, typename ...Args>
   std::unique_ptr<T> make_unique( Args&& ...args )
   {
      return std::unique_ptr<T>( new T( std::forward<Args>(args)... ) );
   }
}
#endif

#include <TNL/Assert.h>
#include <TNL/String.h>
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
      entries.push_back( std::make_unique< ConfigEntry< EntryType > >( name, description, false ) );
      currentEntry = entries.back().get();
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
      entries.push_back( std::make_unique< ConfigEntry< EntryType > >( name, description, true ) );
      currentEntry = entries.back().get();
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
      entries.push_back( std::make_unique< ConfigEntry< EntryType > >( name, description, false, defaultValue ) );
      currentEntry = entries.back().get();
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
      entries.push_back( std::make_unique< ConfigEntryList< EntryType > >( name, description, false ) );
      currentEntry = entries.back().get();
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
      entries.push_back( std::make_unique< ConfigEntryList< EntryType > >( name, description, true ) );
      currentEntry = entries.back().get();
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
      entries.push_back( std::make_unique< ConfigEntryList< EntryType > >( name, description, false, defaultValue ) );
      currentEntry = entries.back().get();
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
      TNL_ASSERT_TRUE( this->currentEntry, "there is no current entry" );
      ConfigEntry< EntryType >& entry = dynamic_cast< ConfigEntry< EntryType >& >( *currentEntry );
      entry.getEnumValues().push_back( entryEnum );
   }

   /**
    * \brief Adds new entry enumeration of type \e char.
    *
    * Adds new option of setting an entry value.
    * \param entryEnum Value of the entry enumeration.
    */
   void addEntryEnum( const char* entryEnum )
   {
      TNL_ASSERT_TRUE( this->currentEntry, "there is no current entry" );
      ConfigEntry< String >& entry = dynamic_cast< ConfigEntry< String >& >( *currentEntry );
      entry.getEnumValues().push_back( String( entryEnum ) );
   }

   /**
    * \brief Adds delimeter/section to the configuration description.
    *
    * \param delimeter String that defines how the delimeter looks like.
    */
   void addDelimiter( const String& delimiter )
   {
      entries.push_back( std::make_unique< ConfigDelimiter >( delimiter ) );
      currentEntry = nullptr;
   }

   /**
    * \brief Gets entry out of the configuration description.
    *
    * \param name Name of the entry.
    */
   const ConfigEntryBase* getEntry( const String& name ) const
   {
      const int entries_num = entries.size();
      for( int i = 0; i < entries_num; i++ )
         if( entries[ i ]->name == name )
            return entries[ i ].get();
      return nullptr;
   }


   //! Returns empty string if given entry does not exist
   //const String getEntryType( const char* name ) const;

   //! Returns zero pointer if there is no default value
   template< class T >
   const T* getDefaultValue( const String& name ) const
   {
      const int entries_num = entries.size();
      for( int i = 0; i < entries_num; i++ )
         if( entries[ i ]->name == name ) {
            if( entries[ i ]->hasDefaultValue ) {
               const ConfigEntry< T >& entry = dynamic_cast< ConfigEntry< T >& >( *entries[ i ] );
               return entry->default_value;
            }
            return nullptr;
         }
      std::cerr << "Asking for the default value of unknown parameter." << std::endl;
      return nullptr;
   }
 
   //! Returns zero pointer if there is no default value
   template< class T >
   T* getDefaultValue( const String& name )
   {
      const int entries_num = entries.size();
      for( int i = 0; i < entries_num; i++ )
         if( entries[ i ] -> name == name ) {
            if( entries[ i ] -> hasDefaultValue ) {
               ConfigEntry< T >& entry = dynamic_cast< ConfigEntry< T >& >( *entries[ i ] );
               return entry->default_value;
            }
            return nullptr;
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

protected:
   std::vector< std::unique_ptr< ConfigEntryBase > > entries;
   ConfigEntryBase* currentEntry = nullptr;
};

} //namespace Config
} //namespace TNL
