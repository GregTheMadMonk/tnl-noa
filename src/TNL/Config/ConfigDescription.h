/***************************************************************************
                          Config::ConfigDescription.h  -  description
                             -------------------
    begin                : 2007/06/09
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iomanip>
#include <string>
#include <vector>
#include <memory>

#include <TNL/Assert.h>
#include <TNL/String.h>
#include <TNL/param-types.h>
#include <TNL/Config/ConfigEntryType.h>
#include <TNL/Config/ConfigEntry.h>
#include <TNL/Config/ConfigEntryList.h>
#include <TNL/Config/ConfigDelimiter.h>
#include <TNL/Config/ParameterContainer.h>

namespace TNL {
namespace Config {

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
      isCurrentEntryList = false;
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
      isCurrentEntryList = false;
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
      isCurrentEntryList = false;
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
      isCurrentEntryList = true;
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
      isCurrentEntryList = true;
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
      isCurrentEntryList = true;
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
      if( isCurrentEntryList ) {
         ConfigEntryList< EntryType >& entry = dynamic_cast< ConfigEntryList< EntryType >& >( *currentEntry );
         entry.getEnumValues().push_back( entryEnum );         
      }
      else {
         ConfigEntry< EntryType >& entry = dynamic_cast< ConfigEntry< EntryType >& >( *currentEntry );
         entry.getEnumValues().push_back( entryEnum );
      }
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
      // ConfigDelimiter has empty name
      if( ! name )
         return nullptr;

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
      // ConfigDelimiter has empty name
      if( ! name )
         return nullptr;

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
      // ConfigDelimiter has empty name
      if( ! name )
         return nullptr;

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
      return nullptr;
   }

   /**
    * \brief Fills in the parameters from the \e parameter_container.
    *
    * Parameters which were not defined in the command line by user but have their default value are added to the congiguration description.
    * If there is missing entry with defined default value in the Config::ParameterContainer it is going to be added.
    * \param parameter_container Name of the ParameterContainer object.
    */
   void addMissingEntries( Config::ParameterContainer& parameter_container ) const
   {
      const int size = entries.size();
      for( int i = 0; i < size; i++ )
      {
         const char* entry_name = entries[ i ]->name.getString();
         if( entries[ i ]->hasDefaultValue &&
             ! parameter_container.checkParameter( entry_name ) )
         {
            if( entries[ i ]->getEntryType() == "String" )
            {
               ConfigEntry< String >& entry = dynamic_cast< ConfigEntry< String >& >( *entries[ i ] );
               parameter_container.addParameter< String >( entry_name, entry.defaultValue );
               continue;
            }
            if( entries[ i ]->getEntryType() == "bool" )
            {
               ConfigEntry< bool >& entry = dynamic_cast< ConfigEntry< bool >& >( *entries[ i ] );
               parameter_container.addParameter< bool >( entry_name, entry.defaultValue );
               continue;
            }
            if( entries[ i ]->getEntryType() == "int" )
            {
               ConfigEntry< int >& entry = dynamic_cast< ConfigEntry< int >& >( *entries[ i ] );
               parameter_container.addParameter< int >( entry_name, entry.defaultValue );
               continue;
            }
            if( entries[ i ]->getEntryType() == "double" )
            {
               ConfigEntry< double >& entry = dynamic_cast< ConfigEntry< double >& >( *entries[ i ] );
               parameter_container.addParameter< double >( entry_name, entry.defaultValue );
               continue;
            }
            
            if( entries[ i ]->getEntryType() == "ConfigEntryList< String >" )
            {
               ConfigEntryList< String >& entry = dynamic_cast< ConfigEntryList< String >& >( *entries[ i ] );
               parameter_container.addList< String >( entry_name, entry.defaultValue );
               continue;
            }
            if( entries[ i ]->getEntryType() == "ConfigEntryList< bool >" )
            {
               ConfigEntryList< bool >& entry = dynamic_cast< ConfigEntryList< bool >& >( *entries[ i ] );
               parameter_container.addList< bool >( entry_name, entry.defaultValue );
               continue;
            }
            if( entries[ i ]->getEntryType() == "ConfigEntryList< int >" )
            {
               ConfigEntryList< int >& entry = dynamic_cast< ConfigEntryList< int >& >( *entries[ i ] );
               parameter_container.addList< int >( entry_name, entry.defaultValue );
               continue;
            }
            if( entries[ i ]->getEntryType() == "ConfigEntryList< double >" )
            {
               ConfigEntryList< double >& entry = dynamic_cast< ConfigEntryList< double >& >( *entries[ i ] );
               parameter_container.addList< double >( entry_name, entry.defaultValue );
               continue;
            }
         }
      }
   }

   //! Check for all entries with the flag 'required'.
   /*! Returns false if any parameter is missing.
    */
   bool checkMissingEntries( Config::ParameterContainer& parameter_container,
                             bool printUsage,
                             const char* programName ) const
   {
      const int size = entries.size();
      std::vector< std::string > missingParameters;
      for( int i = 0; i < size; i++ )
      {
         const char* entry_name = entries[ i ] -> name.getString();
         if( entries[ i ] -> required &&
             ! parameter_container.checkParameter( entry_name ) )
            missingParameters.push_back( entry_name );
      }
      if( missingParameters.size() > 0 )
      {
         std::cerr << "Some mandatory parameters are misssing. They are listed at the end. " << std::endl;
         if( printUsage )
            this->printUsage( programName );
         std::cerr << "Add the following missing parameters to the command line: " << std::endl << "   ";
         for( int i = 0; i < (int) missingParameters.size(); i++ )
            std::cerr << "--" << missingParameters[ i ] << " ... ";
         std::cerr << std::endl;
         return false;
      }
      return true;
   }

   /**
    * \brief Prints configuration description with the \e program_name at the top.
    *
    * \param program_name Name of the program
    */
   void printUsage( const char* program_name ) const
   {
      std::cout << "Usage of: " << program_name << std::endl << std::endl;
      const int entries_num = entries.size();
      int max_name_length( 0 );
      int max_type_length( 0 );
      for( int j = 0; j < entries_num; j++ )
         if( ! entries[ j ]->isDelimiter() )
         {
            max_name_length = std::max( max_name_length,
                        entries[ j ] -> name. getLength() );
            max_type_length = std::max( max_type_length,
                        entries[ j ] -> getUIEntryType().getLength() );
         }
      max_name_length += 2; // this is for '--'

      for( int j = 0; j < entries_num; j++ )
      {
         if( entries[ j ]->isDelimiter() )
         {
            std::cout << std::endl;
            std::cout << entries[ j ]->description;
            std::cout << std::endl << std::endl;
         }
         else
         {
            std::cout << std::setw( max_name_length + 3 ) << String( "--" ) + entries[ j ]->name
                 << std::setw( max_type_length + 5 ) << entries[ j ] -> getUIEntryType()
                 << "    " << entries[ j ]->description;
            if( entries[ j ] -> required )
               std::cout << " *** REQUIRED ***";
            if( entries[ j ]->hasEnumValues() )
            {
               std::cout << std::endl
                    << std::setw( max_name_length + 3 ) << ""
                    << std::setw( max_type_length + 5 ) << ""
                    << "    ";
               entries[ j ]->printEnumValues();
            }
            if( entries[ j ]->hasDefaultValue )
            {
               std::cout << std::endl
                    << std::setw( max_name_length + 3 ) << ""
                    << std::setw( max_type_length + 5 ) << ""
                    << "    ";
               std::cout << "- Default value is: " << entries[ j ]->printDefaultValue();
            }
            std::cout << std::endl;
         }
      }
      std::cout << std::endl;
   }

   //bool parseConfigDescription( const char* file_name );

protected:
   std::vector< std::unique_ptr< ConfigEntryBase > > entries;
   ConfigEntryBase* currentEntry = nullptr;
   bool isCurrentEntryList = false;
};

} //namespace Config
} //namespace TNL

#include <TNL/Config/parseCommandLine.h>
