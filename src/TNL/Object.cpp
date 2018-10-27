/***************************************************************************
                          Object.cpp  -  description
                             -------------------
    begin                : 2005/10/15
    copyright            : (C) 2005 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Object.h>
#include <TNL/Assert.h>
#include <iostream>
#include <fstream>
#include <cstring>

namespace TNL {

const char magic_number[] = "TNLMN";

String Object :: getType()
{
   return String( "Object" );
}

String Object :: getTypeVirtual() const
{
   return this->getType();
}

String Object :: getSerializationType()
{
   return String( "Object" );
}

String Object :: getSerializationTypeVirtual() const
{
   return this->getSerializationType();
}

bool Object :: save( File& file ) const
{
   if( ! file. write( magic_number, strlen( magic_number ) ) )
      return false;
   if( ! this->getSerializationTypeVirtual().save( file ) ) return false;
   return true;
}

bool Object :: load( File& file )
{
   String objectType;
   if( ! getObjectType( file, objectType ) )
      return false;
   if( objectType != this->getSerializationTypeVirtual() )
   {
      std::cerr << "Given file contains instance of " << objectType << " but " << getSerializationTypeVirtual() << " is expected." << std::endl;
      return false;
   }
   return true;
}

bool Object :: boundLoad( File& file )
{
   return load( file );
}

bool Object :: save( const String& fileName ) const
{
   File file;
   if( ! file. open( fileName, IOMode::write ) )
   {
      std::cerr << "I am not able to open the file " << fileName << " for writing." << std::endl;
      return false;
   }
   return this->save( file );
}

bool Object :: load( const String& fileName )
{
   File file;
   if( ! file. open( fileName, IOMode::read ) )
   {
      std::cerr << "I am not able to open the file " << fileName << " for reading." << std::endl;
      return false;
   }
   return this->load( file );
}

bool Object :: boundLoad( const String& fileName )
{
   File file;
   if( ! file. open( fileName, IOMode::read ) )
   {
      std::cerr << "I am not able to open the file " << fileName << " for reading." << std::endl;
      return false;
   }
   return this->boundLoad( file );
}


bool getObjectType( File& file, String& type )
{
   char mn[ 10 ];
   if( ! file. read( mn, strlen( magic_number ) ) )
   {
      std::cerr << "Unable to read file " << file. getFileName() << " ... " << std::endl;
      return false;
   }
   if( strncmp( mn, magic_number, 5 ) != 0 &&
       strncmp( mn, "SIM33", 5 ) != 0 )
   {
       std::cout << "Not a TNL file (wrong magic number)." << std::endl;
       return false;
   }
   if( ! type. load( file ) )
   {
       std::cerr << "Cannot load the object type." << std::endl;
       return false;
   }
   return true;
}

bool getObjectType( const String& fileName, String& type )
{
   File binaryFile;
   if( ! binaryFile. open( fileName, IOMode::read ) )
   {
      std::cerr << "I am not able to open the file " << fileName << " for detecting the object inside!" << std::endl;
      return false;
   }
   return getObjectType( binaryFile, type );
}

bool parseObjectType( const String& objectType,
                      Containers::List< String >& parsedObjectType )
{
   parsedObjectType.reset();
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
   String objectName( objectType. getString(), 0, objectTypeLength - i );
   if( ! parsedObjectType. Append( objectName ) )
      return false;
   i ++;

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
         templateBrackets ++;
      if( ! templateBrackets )
      {
         if( objectType[ i ] == ',' ||
             objectType[ i ] == '>' )
         {
            if( buffer != "" )
            {
               if( ! parsedObjectType. Append( buffer.strip( ' ' ) ) )
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
   return true;
}

} // namespace TNL
