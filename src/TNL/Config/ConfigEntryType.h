/***************************************************************************
                          ConfigEntryType.h  -  description
                             -------------------
    begin                : Jul 5, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

namespace TNL {
namespace Config {

template< typename EntryType >
inline String getUIEntryType() { return "Unknown type."; };

template<> inline String getUIEntryType< String >() { return "string"; };
template<> inline String getUIEntryType< bool >()      { return "bool"; };
template<> inline String getUIEntryType< int >()       { return "integer"; };
template<> inline String getUIEntryType< double >()    { return "real"; };

template<> inline String getUIEntryType< List< String > >() { return "list of string"; };
template<> inline String getUIEntryType< List< bool > >()      { return "list of bool"; };
template<> inline String getUIEntryType< List< int > >()       { return "list of integer"; };
template<> inline String getUIEntryType< List< double > >()    { return "list of real"; };

struct ConfigEntryType
{
   String basic_type;

   bool list_entry;

   ConfigEntryType(){};

   ConfigEntryType( const String& _basic_type,
                     const bool _list_entry )
   : basic_type( _basic_type ),
     list_entry( _list_entry ){}

   void Reset()
   {
      basic_type. setString( 0 );
      list_entry = false;
   };
};

} // namespace Config
} // namespace TNL
