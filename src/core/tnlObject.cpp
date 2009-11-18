/***************************************************************************
                          tnlObject.cpp  -  description
                             -------------------
    begin                : 2005/10/15
    copyright            : (C) 2005 by Tom� Oberhuber
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

#include "tnlObject.h"
#include <assert.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include "debug.h"

const char magic_number[] = "SIM33";
//--------------------------------------------------------------------------
tnlObject :: tnlObject( )
{
}
//--------------------------------------------------------------------------
tnlObject :: tnlObject( const tnlObject& object )
{
}
//--------------------------------------------------------------------------
tnlString tnlObject :: GetType() const
{
   return tnlString( "mObject" );
}
//--------------------------------------------------------------------------
void tnlObject :: SetName( const char* _name )
{
   name. SetString( _name );
}
//--------------------------------------------------------------------------
const tnlString& tnlObject :: GetName() const
{
   return name;
}
//--------------------------------------------------------------------------
bool tnlObject :: Save( ostream& file ) const
{
   dbgFunctionName( "mObject", "Save" );
   dbgCout( "Writing magic number." );
   file. write( magic_number, strlen( magic_number ) * sizeof( char ) ); 
   if( file. bad() ) return false;
   dbgCout( "Writing object name " << name );
   if( ! GetType(). Save( file ) || ! name. Save( file ) ) return false;
   return true;
}
//--------------------------------------------------------------------------
bool tnlObject :: Load( istream& file )
{
   dbgFunctionName( "mObject", "Load" );
   dbgCout( "Reading object type " );
   tnlString load_type;
   if( ! GetObjectType( file, load_type ) )
      return false;
   if( load_type != GetType() )
   {
      cerr << "Given file contains instance of " << load_type << " but " << GetType() << " is expected." << endl;
      return false;
   }
   dbgCout( "Reading object name " );
   if( ! name. Load( file ) ) return false;
   return true;
}
//--------------------------------------------------------------------------
bool GetObjectType( istream& file, tnlString& type )
{
   dbgFunctionName( "", "GetObjectType" );
   dbgCout( "Chacking magic number." );
   char mn[ 10 ];
   file. read( mn, strlen( magic_number ) * sizeof( char ) );
   if( strncmp( mn, magic_number, 5 ) != 0 ) return false;
   if( ! type. Load( file ) ) return false;
   return true;
}
//--------------------------------------------------------------------------
bool GetObjectType( const char* file_name, tnlString& type )
{
   fstream file;
   file. open( file_name, ios :: in | ios :: binary );
   if( ! file )
   {
      cerr << "Unable to open file " << file_name << " ... " << endl;
      return false;
   }
   bool ret_val = GetObjectType( file, type );
   file. close();
   return ret_val;
}
