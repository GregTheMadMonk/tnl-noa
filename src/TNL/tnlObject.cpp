/***************************************************************************
                          tnlObject.cpp  -  description
                             -------------------
    begin                : 2005/10/15
    copyright            : (C) 2005 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/tnlObject.h>
#include <TNL/debug/tnlDebug.h>
#include <TNL/core/tnlAssert.h>
#include <TNL/core/tnlFile.h>
#include <TNL/core/tnlList.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <stdio.h>

namespace TNL {

const char magic_number[] = "TNLMN";

tnlString tnlObject :: getType()
{
   return tnlString( "tnlObject" );
}

tnlString tnlObject :: getTypeVirtual() const
{
   return this->getType();
}

tnlString tnlObject :: getSerializationType()
{
   return tnlString( "tnlObject" );
}

tnlString tnlObject :: getSerializationTypeVirtual() const
{
   return this->getSerializationType();
}

bool tnlObject :: save( tnlFile& file ) const
{
#ifdef HAVE_NOT_CXX11
   if( ! file. write< const char, tnlHost, int >( magic_number, strlen( magic_number ) ) )
#else
   if( ! file. write( magic_number, strlen( magic_number ) ) )
#endif
      return false;
   if( ! this->getSerializationTypeVirtual().save( file ) ) return false;
   return true;
}

bool tnlObject :: load( tnlFile& file )
{
   tnlString objectType;
   if( ! getObjectType( file, objectType ) )
      return false;
   if( objectType != this->getSerializationTypeVirtual() )
   {
      std::cerr << "Given file contains instance of " << objectType << " but " << getSerializationTypeVirtual() << " is expected." << std::endl;
      return false;
   }
   return true;
}

bool tnlObject :: boundLoad( tnlFile& file )
{
   return load( file );
}

bool tnlObject :: save( const tnlString& fileName ) const
{
   tnlFile file;
   if( ! file. open( fileName, tnlWriteMode ) )
   {
      std::cerr << "I am not bale to open the file " << fileName << " for writing." << std::endl;
      return false;
   }
   if( ! this->save( file ) )
      return false;
   if( ! file. close() )
   {
      std::cerr << "An error occurred when I was closing the file " << fileName << "." << std::endl;
      return false;
   }
   return true;
}

bool tnlObject :: load( const tnlString& fileName )
{
   tnlFile file;
   if( ! file. open( fileName, tnlReadMode ) )
   {
      std::cerr << "I am not bale to open the file " << fileName << " for reading." << std::endl;
      return false;
   }
   if( ! this->load( file ) )
      return false;
   if( ! file. close() )
   {
      std::cerr << "An error occurred when I was closing the file " << fileName << "." << std::endl;
      return false;
   }
   return true;
}

bool tnlObject :: boundLoad( const tnlString& fileName )
{
   tnlFile file;
   if( ! file. open( fileName, tnlReadMode ) )
   {
      std::cerr << "I am not bale to open the file " << fileName << " for reading." << std::endl;
      return false;
   }
   if( ! this->boundLoad( file ) )
      return false;
   if( ! file. close() )
   {
      std::cerr << "An error occurred when I was closing the file " << fileName << "." << std::endl;
      return false;
   }
   return true;
}


bool getObjectType( tnlFile& file, tnlString& type )
{
   dbgFunctionName( "", "getObjectType" );
   char mn[ 10 ];
#ifdef HAVE_NOT_CXX11
   if( ! file. read< char, tnlHost, int >( mn, strlen( magic_number ) ) )
#else
   if( ! file. read( mn, strlen( magic_number ) ) )
#endif
   {
      std::cerr << "Unable to read file " << file. getFileName() << " ... " << std::endl;
      return false;
   }
   if( strncmp( mn, magic_number, 5 ) != 0 &&
       strncmp( mn, "SIM33", 5 ) != 0 ) return false;
   if( ! type. load( file ) ) return false;
   return true;
}

bool getObjectType( const tnlString& fileName, tnlString& type )
{
   tnlFile binaryFile;
   if( ! binaryFile. open( fileName, tnlReadMode ) )
   {
      std::cerr << "I am not able to open the file " << fileName << " for detecting the object inside!" << std::endl;
      return false;
   }
   bool ret_val = getObjectType( binaryFile, type );
   binaryFile. close();
   return ret_val;
}

bool parseObjectType( const tnlString& objectType,
                      tnlList< tnlString >& parsedObjectType )
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
   return true;
}

} // namespace TNL
