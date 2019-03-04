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

inline bool Object::save( File& file ) const
{
   if( ! file.write( magic_number, strlen( magic_number ) ) )
      return false;
   file << this->getSerializationTypeVirtual();
   return true;
}

inline bool Object::load( File& file )
{
   String objectType;
   getObjectType( file, objectType );
   if( objectType != this->getSerializationTypeVirtual() )
   {
      std::cerr << "Given file contains instance of " << objectType << " but " << getSerializationTypeVirtual() << " is expected." << std::endl;
      return false;
   }
   return true;
}

inline bool Object::boundLoad( File& file )
{
   return load( file );
}

inline bool Object::save( const String& fileName ) const
{
   File file;
   if( ! file.open( fileName, IOMode::write ) )
   {
      std::cerr << "I am not able to open the file " << fileName << " for writing." << std::endl;
      return false;
   }
   return this->save( file );
}

inline bool Object::load( const String& fileName )
{
   File file;
   if( ! file.open( fileName, IOMode::read ) )
   {
      std::cerr << "I am not able to open the file " << fileName << " for reading." << std::endl;
      return false;
   }
   return this->load( file );
}

inline bool Object::boundLoad( const String& fileName )
{
   File file;
   if( ! file.open( fileName, IOMode::read ) )
   {
      std::cerr << "I am not able to open the file " << fileName << " for reading." << std::endl;
      return false;
   }
   return this->boundLoad( file );
}

inline void getObjectType( File& file, String& type )
{
   char mn[ 10 ];
   file.read( mn, strlen( magic_number ) );
   if( strncmp( mn, magic_number, 5 ) != 0 )
      throw Exceptions::NotTNLFile();
   file >> type;
}

inline void getObjectType( const String& fileName, String& type )
{
   File binaryFile;
   binaryFile.open( fileName, IOMode::read );
   getObjectType( binaryFile, type );
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

} // namespace TNL
