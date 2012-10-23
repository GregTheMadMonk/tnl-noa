/***************************************************************************
                          tnlObject.cpp  -  description
                             -------------------
    begin                : 2005/10/15
    copyright            : (C) 2005 by Tomas Oberhuber
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

#include <core/tnlObject.h>
#include <debug/tnlDebug.h>
#include <core/tnlAssert.h>
#include <core/tnlFile.h>
#include <core/tnlList.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <stdio.h>

const char magic_number[] = "SIM33";


tnlObject :: tnlObject( const tnlString& _name )
: name( _name )
{
   dbgFunctionName( "tnlObject", "tnlObject" );
   dbgCout( "Initiating object " << getName() );
}

tnlString tnlObject :: getType() const
{
   return tnlString( "tnlObject" );
}

const tnlString& tnlObject :: getName() const
{
   return name;
}

bool tnlObject :: save( tnlFile& file ) const
{
   dbgFunctionName( "tnlObject", "Save" );
   dbgCout( "Writing magic number." );
   if( ! file. write( magic_number, strlen( magic_number ) ) )
      return false;
   dbgCout( "Writing object name " << name );
   if( ! this -> getType(). save( file ) || ! name. save( file ) ) return false;
   return true;
}

bool tnlObject :: load( tnlFile& file )
{
   dbgFunctionName( "tnlObject", "Load" );
   dbgCout( "Reading object type " );
   tnlString load_type;
   if( ! getObjectType( file, load_type ) )
      return false;
   if( load_type != getType() )
   {
      cerr << "Given file contains instance of " << load_type << " but " << getType() << " is expected." << endl;
      return false;
   }
   dbgCout( "Reading object name " );
   if( ! name. load( file ) ) return false;
   return true;
}

bool tnlObject :: save( const tnlString& fileName ) const
{
   tnlFile file;
   if( ! file. open( fileName, tnlWriteMode ) )
   {
      cerr << "I am not bale to open the file " << fileName << " for writing." << endl;
      return false;
   }
   if( ! this -> save( file ) )
      return false;
   if( ! file. close() )
   {
      cerr << "An error occured when I was closing the file " << fileName << "." << endl;
      return false;
   }
   return true;
}

bool tnlObject :: load( const tnlString& fileName )
{
   tnlFile file;
   if( ! file. open( fileName, tnlReadMode ) )
   {
      cerr << "I am not bale to open the file " << fileName << " for reading." << endl;
      return false;
   }
   if( ! this -> load( file ) )
      return false;
   if( ! file. close() )
   {
      cerr << "An error occured when I was closing the file " << fileName << "." << endl;
      return false;
   }
   return true;
}

bool getObjectType( tnlFile& file, tnlString& type )
{
   dbgFunctionName( "", "getObjectType" );
   dbgCout( "Checking magic number." );
   char mn[ 10 ];
   if( ! file. read( mn, strlen( magic_number ) ) )
   {
      cerr << "Unable to read file " << file. getFileName() << " ... " << endl;
      return false;
   }
   if( strncmp( mn, magic_number, 5 ) != 0 ) return false;
   if( ! type. load( file ) ) return false;
   return true;
}

bool getObjectType( const tnlString& fileName, tnlString& type )
{
   tnlFile binaryFile;
   if( ! binaryFile. open( fileName, tnlReadMode ) )
   {
      cerr << "I am not able to open the file " << fileName << " for detecting the object inside!" << endl;
      return false;
   }
   bool ret_val = getObjectType( binaryFile, type );
   binaryFile. close();
   return ret_val;
}

bool parseObjectType( const tnlString& objectType,
                      tnlList< tnlString >& parsedObjectType )
{
   parsedObjectType. EraseAll();
   int objectTypeLength = objectType. getLength();
   int i = 0;
   /****
    * The object type consists of the following:
    * objectName< param1, param2, param3, .. >.
    * We first extract the objectName by finding the first
    * character '<'.
    */
   while( i < objectTypeLength && objectType[ i ] != '<' )
      i ++;
   tnlString objectName( objectType. getString(), 0, objectTypeLength - i );
   if( ! parsedObjectType. Append( objectName ) )
      return false;
   i ++;

   /****
    * Now, we will extract the parameters.
    * Each parameter can be template, so we must compute and pair
    * '<' with '>'.
    */
   int templateBrackets( 0 );
   tnlString buffer( "" );

   while( i < objectTypeLength )
   {
      if( objectType[ i ] == '<' )
         templateBrackets ++;
      if( ! templateBrackets )
      {
         if( objectType[ i ] == ' ' ||
             objectType[ i ] == ',' ||
             objectType[ i ] == '>' )
         {
            if( buffer != "" )
            {
               if( ! parsedObjectType. Append( buffer ) )
                  return false;
               buffer. setString( "" );
            }
         }
         else buffer += objectType[ i ];
      }
      else buffer += objectType[ i ];
      if( objectType[ i ] == '>' )
         templateBrackets --;
      i ++;
   }
}


