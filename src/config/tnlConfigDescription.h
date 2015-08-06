/***************************************************************************
                          tnlConfigDescription.h  -  description
                             -------------------
    begin                : 2007/06/09
    copyright            : (C) 2007 by Tomas Oberhuber
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

#ifndef tnlConfigDescriptionH
#define tnlConfigDescriptionH

#include <core/tnlString.h>
#include <core/tnlList.h>
#include <core/param-types.h>
#include <config/tnlConfigEntryType.h>
#include <config/tnlConfigEntry.h>
#include <config/tnlConfigEntryList.h>
#include <config/tnlConfigDelimiter.h>

class tnlParameterContainer;

class tnlConfigDescription
{
   public:

   tnlConfigDescription();

   template< typename EntryType >
   void addEntry( const tnlString& name,
                  const tnlString& description )
   {
      currentEntry = new tnlConfigEntry< EntryType >( name, description, false );
      entries.Append( currentEntry );
   }

   template< typename EntryType >
   void addRequiredEntry( const tnlString& name,
                          const tnlString& description )
   {
      currentEntry = new tnlConfigEntry< EntryType >( name, description, true );
      entries.Append( currentEntry );
   }
   
   template< typename EntryType >
   void addEntry( const tnlString& name,
                  const tnlString& description,
                  const EntryType& defaultValue )
   {
      currentEntry = new tnlConfigEntry< EntryType >( name,
                                                      description,
                                                      false,
                                                      defaultValue );
      entries. Append( currentEntry );
   }

   template< typename EntryType >
   void addList( const tnlString& name,
                 const tnlString& description )
   {
      currentEntry = new tnlConfigEntryList< EntryType >( name, description, false );
      entries.Append( currentEntry );
   }

   template< typename EntryType >
   void addRequiredList( const tnlString& name,
                         const tnlString& description )
   {
      currentEntry = new tnlConfigEntryList< EntryType >( name, description, true );
      entries.Append( currentEntry );
   }

   template< typename EntryType >
   void addList( const tnlString& name,
                 const tnlString& description,
                 const EntryType& defaultValue )
   {
      currentEntry = new tnlConfigEntryList< EntryType >( name,
                                                          description,
                                                          false,
                                                          defaultValue );
      entries. Append( currentEntry );
   }

   template< typename EntryType >
   void addEntryEnum( const EntryType& entryEnum )
   {
      tnlAssert( this->currentEntry,);
      ( ( tnlConfigEntry< EntryType >* ) currentEntry )->getEnumValues().Append( entryEnum );
   }

   void addEntryEnum( const char* entryEnum )
   {
      tnlAssert( this->currentEntry,);
      ( ( tnlConfigEntry< tnlString >* ) currentEntry )->getEnumValues().Append( tnlString( entryEnum ) );
   }

   void addDelimiter( const tnlString& delimiter )
   {
      entries.Append( new tnlConfigDelimiter( delimiter ) );
      currentEntry = 0;
   }

   const tnlConfigEntryBase* getEntry( const tnlString& name ) const
   {
      for( int i = 0; i < entries.getSize(); i++ )
         if( entries[ i ]->name == name )
            return entries[ i ];
      return NULL;
   }

   
   //! Returns empty string if given entry does not exist
   //const tnlString getEntryType( const char* name ) const;

   //! Returns zero pointer if there is no default value
   template< class T > const T* getDefaultValue( const tnlString& name ) const
   {
      int i;
      const int entries_num = entries. getSize();
      for( i = 0; i < entries_num; i ++ )
         if( entries[ i ] -> name == name )
         {
            if( entries[ i ] -> hasDefaultValue )
               return ( ( tnlConfigEntry< T > * ) entries[ i ] ) -> default_value;
            else return NULL;
         }
      cerr << "Asking for the default value of uknown parameter." << endl;
      return NULL;
   }
   
   //! Returns zero pointer if there is no default value
   template< class T > T* getDefaultValue( const tnlString& name )
   {
      int i;
      const int entries_num = entries. getSize();
      for( i = 0; i < entries_num; i ++ )
         if( entries[ i ] -> name == name )
         {
            if( entries[ i ] -> hasDefaultValue )
               return ( ( tnlConfigEntry< T > * ) entries[ i ] ) -> default_value;
            else return NULL;
         }
      cerr << "Asking for the default value of uknown parameter." << endl;
      return NULL;
   }

   //! If there is missing entry with defined default value in the tnlParameterContainer it is going to be added
   void addMissingEntries( tnlParameterContainer& parameter_container ) const;

   //! Check for all entries with the flag 'required'.
   /*! Returns false if any parameter is missing.
    */
   bool checkMissingEntries( tnlParameterContainer& parameter_container,
                             bool printUsage,
                             const char* programName ) const;

   void printUsage( const char* program_name ) const;

   //bool parseConfigDescription( const char* file_name );

   ~tnlConfigDescription();

   protected:

   tnlList< tnlConfigEntryBase* > entries;

   tnlConfigEntryBase* currentEntry;

};


#endif
