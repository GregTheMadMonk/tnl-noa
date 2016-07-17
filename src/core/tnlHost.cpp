/***************************************************************************
                          tnlHost.h  -  description
                             -------------------
    begin                : Nov 7, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

#include <core/tnlHost.h>
#ifdef HAVE_OPENMP
#include <omp.h>
#endif
#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>

namespace TNL {


bool tnlHost::ompEnabled( true );
int tnlHost::maxThreadsCount( -1 );

tnlString tnlHost::getDeviceType()
{
   return tnlString( "tnlHost" );
};

size_t tnlHost::getFreeMemory()
{
   long pages = sysconf(_SC_PHYS_PAGES);
   long page_size = sysconf(_SC_PAGE_SIZE);
   return pages * page_size;
};

void tnlHost::enableOMP()
{
   ompEnabled = true;
}

void tnlHost::disableOMP()
{
   ompEnabled = false;
}

void tnlHost::setMaxThreadsCount( int maxThreadsCount_ )
{
   maxThreadsCount = maxThreadsCount_;
#ifdef HAVE_OPENMP
   omp_set_num_threads( maxThreadsCount );
#endif
}

int tnlHost::getMaxThreadsCount()
{
#ifdef HAVE_OPENMP
   if( maxThreadsCount == -1 )
      return omp_get_max_threads();
   return maxThreadsCount;
#else
   return 0;
#endif
}
 
int tnlHost::getThreadIdx()
{
#ifdef HAVE_OPENMP
   return omp_get_thread_num();
#else
   return 0;
#endif
}

void tnlHost::configSetup( tnlConfigDescription& config, const tnlString& prefix )
{
#ifdef HAVE_OPENMP
   config.addEntry< bool >( prefix + "omp-enabled", "Enable support of OpenMP.", true );
   config.addEntry<  int >( prefix + "omp-max-threads", "Set maximum number of OpenMP threads.", omp_get_max_threads() );
#else
   config.addEntry< bool >( prefix + "omp-enabled", "Enable support of OpenMP (not supported on this system).", false );
   config.addEntry<  int >( prefix + "omp-max-threads", "Set maximum number of OpenMP threads (not supported on this system).", 0 );
#endif
 
}
 
bool tnlHost::setup( const tnlParameterContainer& parameters,
                    const tnlString& prefix )
{
   ompEnabled = parameters.getParameter< bool >( prefix + "omp-enabled" );
   maxThreadsCount = parameters.getParameter< int >( prefix + "omp-max-threads" );
   return true;
}

} // namespace TNL

