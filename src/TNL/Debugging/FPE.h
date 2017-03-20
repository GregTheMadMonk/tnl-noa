/***************************************************************************
                          MeshConfigBase.h  -  description
                             -------------------
    begin                : Nov 6, 2016
    copyright            : (C) 2016 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <cfenv>
#include <signal.h>

#include <TNL/Debugging/StackBacktrace.h>

namespace TNL {
namespace Debugging {

static void
printStackBacktraceAndAbort( int sig = 0 )
{
   if( sig == SIGSEGV )
      fprintf(stderr, "Invalid memory reference, printing backtrace and aborting...\n");
   else if( sig == SIGFPE ) {
      /*
       * Unfortunately it is not possible to get the floating-point exception type
       * from a signal handler. Otherwise, it would be done this way:
       *
       *    fprintf(stderr, "Floating-point exception");
       *    if(fetestexcept(FE_DIVBYZERO))  fprintf(stderr, " FE_DIVBYZERO");
       *    if(fetestexcept(FE_INEXACT))    fprintf(stderr, " FE_INEXACT");
       *    if(fetestexcept(FE_INVALID))    fprintf(stderr, " FE_INVALID");
       *    if(fetestexcept(FE_OVERFLOW))   fprintf(stderr, " FE_OVERFLOW");
       *    if(fetestexcept(FE_UNDERFLOW))  fprintf(stderr, " FE_UNDERFLOW");
       *    fprintf(stderr, " occurred, printing backtrace and aborting...\n");
       */
      fprintf(stderr, "Floating-point exception occurred, printing backtrace and aborting...\n");
   }
   else
      fprintf( stderr, "Aborting due to signal %d...\n", sig );
   printStackBacktrace();
   abort();
}

/*
 * Registers handler for SIGSEGV and SIGFPE signals and enables conversion of
 * floating-point exceptions into SIGFPE. This is useful e.g. for tracing where
 * NANs occurred. Example usage:
 *
 * int main()
 * {
 *    #ifndef NDEBUG
 *       registerFloatingPointExceptionTracking()
 *    #endif
 *    [start some computation here...]
 * }
 */
static void
trackFloatingPointExceptions()
{
   signal( SIGSEGV, printStackBacktraceAndAbort );
   signal( SIGFPE,  printStackBacktraceAndAbort );
   feenableexcept( FE_ALL_EXCEPT & ~FE_INEXACT );
}

} // namespace Debugging
} // namespace TNL
