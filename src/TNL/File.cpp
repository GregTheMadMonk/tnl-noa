/***************************************************************************
                          File.cpp  -  description
                             -------------------
    begin                : Oct 22, 2010
    copyright            : (C) 2010 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/File.h>

namespace TNL {

File :: File()
: mode( IOMode::undefined ),
  file( NULL ),
  fileOK( false ),
  writtenElements( 0 ),
  readElements( 0 )
{
}

File :: ~File()
{
   // destroying a file without closing is a memory leak
   // (an open file descriptor is left behind, on Linux there is typically
   // only a limited number of descriptors available to each process)
   close();
}

bool File :: open( const String& fileName,
                   const IOMode mode )
{
   // close the existing file to avoid memory leaks
   this->close();

   this->fileName = fileName;
   if( mode == IOMode::read )
      file = std::fopen( fileName.getString(), "rb" );
   if( mode == IOMode::write )
      file = std::fopen( fileName.getString(), "wb" );
   if( file ==  NULL )
   {
      std::cerr << "I am not able to open the file " << fileName << ". ";
      std::perror( "" );
      return false;
   }
   this->fileOK = true;
   this->mode = mode;
   return true;
}

bool File :: close()
{
   if( file && std::fclose( file ) != 0 )
   {
      std::cerr << "I was not able to close the file " << fileName << " properly!" << std::endl;
      return false;
   }
   // reset all attributes
   mode = IOMode::undefined;
   file = NULL;
   fileOK = false;
   fileName = "";
   readElements = writtenElements = 0;
   return true;
}

bool fileExists( const String& fileName )
{
  std::fstream file;
  file.open( fileName.getString(), std::ios::in );
  return ! file.fail();
}

} // namespace TNL
