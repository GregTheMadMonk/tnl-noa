/***************************************************************************
                          OutputRedirection.h  -  description
                             -------------------
    begin                : Jul 23, 2020
    copyright            : (C) 2005 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include <unistd.h>
#include <iostream>

namespace TNL {
namespace Debugging {

class OutputRedirection
{
   int backupFd = -1;
   int targetFd = -1;
   FILE* file = nullptr;

public:
   OutputRedirection() = delete;

   OutputRedirection(int targetFd) : targetFd(targetFd) {}

   bool redirect(std::string fname)
   {
      // restore the original stream if there is any backup
      if( backupFd >= 0 || file )
         if( ! restore() )
             return false;

      // first open the file
      file = ::fopen(fname.c_str(), "w");
      if( file == nullptr ) {
         std::cerr << "error: fopen() failed, output is not redirected." << std::endl;
         return false;
      }

      // then backup the original file descriptors
      backupFd = ::dup(targetFd);
      if( backupFd < 0 ) {
         std::cerr << "error: dup() failed, output is not redirected." << std::endl;
         return false;
      }

      // finally redirect stdout and stderr
      if( ::dup2(::fileno(file), targetFd) < 0 ) {
         std::cerr << "error: dup2() failed, output is not redirected." << std::endl;
         return false;
      }

      return true;
   }

   bool restore()
   {
      // first restore the original file descriptor
      if( backupFd >= 0 ) {
         if( ::dup2(backupFd, targetFd) < 0 ) {
            std::cerr << "error: dup2() failed, output is not restored." << std::endl;
            return false;
         }
         backupFd = -1;
      }

      // then close the file
      if( file != nullptr ) {
         ::fclose(file);
         file = nullptr;
      }
      return true;
   }

   ~OutputRedirection()
   {
       restore();
   }
};

inline bool
redirect_stdout_stderr(std::string stdout_fname, std::string stderr_fname, bool restore = false)
{
   static OutputRedirection stdoutRedir( STDOUT_FILENO );
   static OutputRedirection stderrRedir( STDERR_FILENO );

   if( restore == false ) {
      if( ! stdoutRedir.redirect(stdout_fname) )
         return false;
      if( ! stderrRedir.redirect(stderr_fname) )
         return false;
   }
   else {
      stdoutRedir.restore();
      stderrRedir.restore();
   }

   return true;
}

} // namespace Debugging
} // namespace TNL