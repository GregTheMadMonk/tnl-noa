/***************************************************************************
                          Object_impl.h  -  description
                             -------------------
    begin                : 2005/10/15
    copyright            : (C) 2005 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iostream>
#include <fstream>
#include <cstring>

#include <TNL/Object.h>
#include <TNL/Exceptions/NotTNLFile.h>
#include <TNL/Exceptions/ObjectTypeMismatch.h>

namespace TNL {

static constexpr char magic_number[] = "TNLMN";

inline String Object::getType()
{
   return String( "Object" );
}

inline String Object::getTypeVirtual() const
{
   return this->getType();
}

inline String Object::getSerializationType()
{
   return String( "Object" );
}

inline String Object::getSerializationTypeVirtual() const
{
   return this->getSerializationType();
}

inline void Object::save( File& file ) const
{
   file.save( magic_number, strlen( magic_number ) );
   file << this->getSerializationTypeVirtual();
}

inline void Object::load( File& file )
{
   String objectType = getObjectType( file );
   if( objectType != this->getSerializationTypeVirtual() )
      throw Exceptions::ObjectTypeMismatch( this->getSerializationTypeVirtual(), objectType );
}

inline void Object::boundLoad( File& file )
{
   this->load( file );
}

inline void Object::save( const String& fileName ) const
{
   File file;
   file.open( fileName, File::Mode::Out );
   this->save( file );
}

inline void Object::load( const String& fileName )
{
   File file;
   file.open( fileName, File::Mode::In );
   this->load( file );
}

inline void Object::boundLoad( const String& fileName )
{
   File file;
   file.open( fileName, File::Mode::In );
   this->boundLoad( file );
}

inline String getObjectType( File& file )
{
   char mn[ 10 ];
   String type;
   file.load( mn, strlen( magic_number ) );
   if( strncmp( mn, magic_number, 5 ) != 0 )
      throw Exceptions::NotTNLFile();
   file >> type;
   return type;
}

inline String getObjectType( const String& fileName )
{
   File binaryFile;
   binaryFile.open( fileName, File::Mode::In );
   return getObjectType( binaryFile );
}

inline std::vector< String >
parseObjectType( const String& objectType )
{
   std::vector< String > parsedObjectType;
   const int objectTypeLength = objectType.getLength();
   int i = 0;
   /****
    * The object type consists of the following:
    * objectName< param1, param2, param3, .. >.
    * We first extract the objectName by finding the first
    * character '<'.
    */
   while( i < objectTypeLength && objectType[ i ] != '<' )
      i++;
   String objectName = objectType.substr( 0, i );
   parsedObjectType.push_back( objectName );
   i++;

   /****
    * Now, we will extract the parameters.
    * Each parameter can be template, so we must count and pair
    * '<' with '>'.
    */
   int templateBrackets( 0 );
   String buffer( "" );

   while( i < objectTypeLength )
   {
      if( objectType[ i ] == '<' )
         templateBrackets++;
      if( ! templateBrackets )
      {
         if( objectType[ i ] == ',' ||
             objectType[ i ] == '>' )
         {
            if( buffer != "" )
            {
               parsedObjectType.push_back( buffer.strip( ' ' ) );
               buffer.clear();
            }
         }
         else buffer += objectType[ i ];
      }
      else buffer += objectType[ i ];
      if( objectType[ i ] == '>' )
         templateBrackets--;
      i++;
   }

   return parsedObjectType;
}

inline void saveHeader( File& file, const String& type )
{
   file.save( magic_number, strlen( magic_number ) );
   file << type;
}

inline void loadHeader( File& file, String& type )
{
   char mn[ 10 ];
   file.load( mn, strlen( magic_number ) );
   if( strncmp( mn, magic_number, 5 ) != 0 )
      throw Exceptions::NotTNLFile();
   file >> type;
}



} // namespace TNL
