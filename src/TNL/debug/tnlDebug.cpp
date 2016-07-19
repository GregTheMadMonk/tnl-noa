/***************************************************************************
                          tnlDebug.cpp  -  description
                             -------------------
    begin                : 2004/09/05
    copyright            : (C) 2004 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <string>
#include <fstream>
#include "tnlDebugParser.h"
#include "tnlDebugStructure.h"
#include "tnlDebug.h"

using namespace std;

int __tnldbg_mpi_i_proc;

static tnlDebugStructure __tnl_debug_structure;
//--------------------------------------------------------------------------
bool tnlInitDebug( const char* file_name, const char* program_name )
{
   std::cout << "tnlDebug initiation..." << std::endl;
   __tnl_debug_structure. setDebug( true );
   tnlDebugParser debug_parser;
   std::fstream in_file;
   in_file. open( file_name, std::ios::in );
   if( ! in_file )
   {
      std::cerr << "Unable to open file " << file_name << std::endl;
      return false;
   }
   debug_parser. setScanner( &in_file );
   int errs = debug_parser. runParsing( &__tnl_debug_structure );
   if( errs != 0 )
   {
      std::cerr << errs << " errors occurred while parsing " << file_name << std::endl;
      return false;
   }
   std::cout << "Parsing of " << file_name << " successful done ..." << std::endl;
#ifdef DEBUG
   std::cout << "Parsed data are ... " << std::endl;
   __tnl_debug_structure. Print();
   std::cout << "---------- end of listing -----------" << std::endl;
#endif
   return true;
}
//--------------------------------------------------------------------------
bool _tnldbg_debug_func( const char* group_name,
                         const char* function_name )
{
   bool debug = __tnl_debug_structure. Debug( group_name, function_name );
   //cout << "Debug ( " << group_name << ", " << function_name
   //     << " ) -> " << debug << std::endl;
   return debug;
}
//--------------------------------------------------------------------------
bool _tnldbg_interactive_func( const char* group_name,
                               const char* function_name )
{
   return __tnl_debug_structure. Interactive( group_name, function_name );
}
