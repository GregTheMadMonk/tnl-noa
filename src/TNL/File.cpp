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

int File :: verbose = 0;

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
   this->fileName = fileName;
   if( verbose )
   {
      std::cout << "Opening file " << fileName;
      if( mode == IOMode::read )
         std::cout << " for reading... " << std::endl;
      else
         std::cout << " for writing ... " << std::endl;
   }
   if( mode == IOMode::read )
      file = fopen( fileName. getString(), "r" );
   if( mode == IOMode::write )
      file = fopen( fileName. getString(), "w" );
   if( file ==  NULL )
   {
      std::cerr << "I am not able to open the file " << fileName << ". ";
      perror( "" );
      return false;
   }
   this->fileOK = true;
   this->mode = mode;
   return true;
}

bool File :: close()
{
   if( verbose )
      std::cout << "Closing the file " << getFileName() << " ... " << std::endl;

   if( fclose( file ) != 0 )
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
};

bool fileExists( const String& fileName )
{
  std::fstream file;
  file.open( fileName. getString(), std::ios::in );
  bool result( true );
  if( ! file )
     result = false;
  file.close();
  return result;
};

} // namespace TNL
