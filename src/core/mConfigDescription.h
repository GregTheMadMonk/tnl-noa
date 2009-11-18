/***************************************************************************
                          mConfigDescription.h  -  description
                             -------------------
    begin                : 2007/06/09
    copyright            : (C) 2007 by Tomá¹ Oberhuber
    email                : oberhuber@seznam.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef mConfigDescriptionH
#define mConfigDescriptionH

#include "tnlString.h"
#include "mList.h"

class mParameterContainer;

struct mConfigGroup
{
   tnlString name;

   tnlString comment;

   mConfigGroup( const char* _name, 
                 const char* _comment )
      : name( _name ),
        comment( _comment ){};
};

struct mConfigEntryType
{
   tnlString basic_type;

   bool list_entry;

   mConfigEntryType(){};
   
   mConfigEntryType( const tnlString& _basic_type,
                     const bool _list_entry )
   : basic_type( _basic_type ),
     list_entry( _list_entry ){}

   void Reset()
   {
      basic_type. SetString( 0 );
      list_entry = false;
   };
};

struct mConfigEntryBase
{
   tnlString name;

   mConfigEntryType type;

   tnlString group;

   tnlString comment;

   bool required;

   bool has_default_value;

   mConfigEntryBase( const char* _name,
                     const mConfigEntryType& _type,
                     const char* _group,
                     const char* _comment,
                     bool _required )
      : name( _name ),
        type( _type ),
        group( _group ),
        comment( _comment ),
        required( _required ),
        has_default_value( false ){};

};

template< class T > struct mConfigEntry : public mConfigEntryBase
{
   T default_value;

   public:
   mConfigEntry( const char* _name,
                 const mConfigEntryType& _type,
                 const char* _group,
                 const char* _description,
                 const T& _default_value )
      : mConfigEntryBase( _name,
                          _type,
                          _group,
                          _description,
                          false ),
        default_value( _default_value ) 
      {
         has_default_value = true;
      };

};

//! Class containing description of the configuration parameters
class mConfigDescription
{
   public:

   mConfigDescription();

   void AddGroup( const char* name,
                  const char* description );

   void AddEntry( const char* name,
                  const mConfigEntryType& type,
                  const char* group,
                  const char* comment,
                  bool required );
   
   template< class T > void AddEntryWithDefaultValue( const char* name,
                                                      const mConfigEntryType& type,
                                                      const char* group,
                                                      const char* comment,
                                                      const T& default_value )
   {
      entries. Append( new mConfigEntry< T >( name,
                                              type,
                                              group,
                                              comment,
                                              default_value ) );
   };

   
   //! Returns zero if given entry does not exist
   const mConfigEntryType* GetEntryType( const char* name ) const;

   //! Returns zero pointer if there is no default value
   template< class T > const T* GetDefaultValue( const char* name ) const
   {
      int i;
      const int entries_num = entries. Size();
      for( i = 0; i < entries_num; i ++ )
         if( entries[ i ] -> name == name )
            if( entries[ i ] -> has_default_value )
               return ( ( mConfigEntry< T > * ) entries[ i ] ) -> default_value;
            else return NULL;
      cerr << "Asking for the default value of uknown parameter." << endl;
      return NULL;
   };
   
   //! Returns zero pointer if there is no default value
   template< class T > T* GetDefaultValue( const char* name )
   {
      int i;
      const int entries_num = entries. Size();
      for( i = 0; i < entries_num; i ++ )
         if( entries[ i ] -> name == name )
            if( entries[ i ] -> has_default_value )
               return ( ( mConfigEntry< T > * ) entries[ i ] ) -> default_value;
            else return NULL;
      cerr << "Asking for the default value of uknown parameter." << endl;
      return NULL;
   };

   //! If there is missing entry with defined default value in the mParameterContainer it is going to be added
   void AddMissingEntries( mParameterContainer& parameter_container ) const;

   //! Check for all entries with the flag 'required'.
   /*! Returns false if any parameter is missing.
    */
   bool CheckMissingEntries( mParameterContainer& parameter_container ) const;

   void PrintUsage( const char* program_name );

   bool ParseConfigDescription( const char* file_name );

   ~mConfigDescription();

   protected:

   mList< mConfigGroup* > groups;

   mList< mConfigEntryBase* > entries;


};


#endif
