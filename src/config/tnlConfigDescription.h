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

class tnlParameterContainer;

template< typename EntryType >
inline tnlString getUIEntryType() { return "Unknown type."; };

template<> inline tnlString getUIEntryType< tnlString >() { return "string"; };
template<> inline tnlString getUIEntryType< bool >()      { return "bool"; };
template<> inline tnlString getUIEntryType< int >()       { return "integer"; };
template<> inline tnlString getUIEntryType< double >()    { return "real"; };

template<> inline tnlString getUIEntryType< tnlList< tnlString > >() { return "list of string"; };
template<> inline tnlString getUIEntryType< tnlList< bool > >()      { return "list of bool"; };
template<> inline tnlString getUIEntryType< tnlList< int > >()       { return "list of integer"; };
template<> inline tnlString getUIEntryType< tnlList< double > >()    { return "list of real"; };

struct tnlConfigEntryType
{
   tnlString basic_type;

   bool list_entry;

   tnlConfigEntryType(){};
   
   tnlConfigEntryType( const tnlString& _basic_type,
                     const bool _list_entry )
   : basic_type( _basic_type ),
     list_entry( _list_entry ){}

   void Reset()
   {
      basic_type. setString( 0 );
      list_entry = false;
   };
};

struct tnlConfigEntryBase
{
   tnlString name;

   tnlString description;

   bool required;

   bool hasDefaultValue;

   tnlConfigEntryBase( const char* name,
                       const char* description,
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

template< class EntryType >
struct tnlConfigEntry : public tnlConfigEntryBase
{
   EntryType defaultValue;

   tnlList< EntryType > enumValues;

   public:

   tnlConfigEntry( const char* name,
                   const char* description,
                   bool required )
      : tnlConfigEntryBase( name,
                            description,
                            required ) 
      {
         hasDefaultValue = false;
      }

   tnlConfigEntry( const char* name,
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
      return ::getParameterType< EntryType >();
   }

   tnlString getUIEntryType() const
   {
      return ::getUIEntryType< EntryType >();
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
            cerr << "The value " << value << " is not allowed for the config entry " << this->name << "." << endl;
            this->printEnumValues();
            return false;
         }
      }
      return true;
   };
};

struct tnlConfigDelimiter : public tnlConfigEntryBase
{
   tnlConfigDelimiter( const char* delimiter )
   : tnlConfigEntryBase( "", delimiter, false )
   {
   };

   bool isDelimiter() const { return true; };
   
   tnlString getEntryType() const { return ""; };

   tnlString getUIEntryType() const { return ""; };
};

//! Class containing description of the configuration parameters
class tnlConfigDescription
{
   public:

   tnlConfigDescription();

   template< typename EntryType >
   void addEntry( const char* name,
                  const char* description )
   {
      currentEntry = new tnlConfigEntry< EntryType >( name, description, false );
      entries.Append( currentEntry );
   }

   template< typename EntryType >
   void addRequiredEntry( const char* name,
                          const char* description )
   {
      currentEntry = new tnlConfigEntry< EntryType >( name, description, true );
      entries.Append( currentEntry );
   }
   
   template< typename EntryType >
   void addEntry( const char* name,
                  const char* description,
                  const EntryType& defaultValue )
   {
      currentEntry = new tnlConfigEntry< EntryType >( name,
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

   void addDelimiter( const char* delimiter )
   {
      entries.Append( new tnlConfigDelimiter( delimiter ) );
      currentEntry = 0;
   }

   const tnlConfigEntryBase* getEntry( const char* name ) const
   {
      for( int i = 0; i < entries.getSize(); i++ )
         if( entries[ i ]->name == name )
            return entries[ i ];
      return NULL;
   }

   
   //! Returns empty string if given entry does not exist
   //const tnlString getEntryType( const char* name ) const;

   //! Returns zero pointer if there is no default value
   template< class T > const T* getDefaultValue( const char* name ) const
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
   template< class T > T* getDefaultValue( const char* name )
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
   bool checkMissingEntries( tnlParameterContainer& parameter_container ) const;

   void printUsage( const char* program_name );

   bool parseConfigDescription( const char* file_name );

   ~tnlConfigDescription();

   protected:

   tnlList< tnlConfigEntryBase* > entries;

   tnlConfigEntryBase* currentEntry;


};


#endif
