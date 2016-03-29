/***************************************************************************
                          tnlFile.cpp  -  description
                             -------------------
    begin                : Oct 22, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
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

#include <core/tnlFile.h>

int tnlFile :: verbose = 0;

tnlFile :: tnlFile()
: mode( tnlUndefinedMode ),
  file( NULL ),
  fileOK( false ),
  writtenElements( 0 ),
  readElements( 0 )
{
}

bool tnlFile :: open( const tnlString& fileName,
                      const tnlIOMode mode )
{
   this->fileName = fileName;
   if( verbose )
   {
      cout << "Opening file " << fileName;
      if( mode == tnlReadMode )
         cout << " for reading... " << endl;
      else
         cout << " for writing ... " << endl;
   }
   if( mode == tnlReadMode )
      file = fopen( fileName. getString(), "r" );
   if( mode == tnlWriteMode )
      file = fopen( fileName. getString(), "w" );
   if( file ==  NULL )
   {
      cerr << "I am not able to open the file " << fileName << ". ";
      perror( "" );
      return false;
   }
   this->fileOK = true;
   this->mode = mode;
   return true;
}

bool tnlFile :: close()
{
   if( verbose )
      cout << "Closing the file " << getFileName() << " ... " << endl;

   if( fclose( file ) != 0 )
   {
      cerr << "I was not able to close the file " << fileName << " properly!" << endl;
      return false;
   }
   readElements = writtenElements = 0;
   return true;
};

bool fileExists( const tnlString& fileName )
{
  fstream file;
  file.open( fileName. getString(), ios::in );
  bool result( true );
  if( ! file )
     result = false;
  file.close();
  return result;
};

