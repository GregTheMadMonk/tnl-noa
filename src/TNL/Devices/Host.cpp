/***************************************************************************
                          Host.h  -  description
                             -------------------
    begin                : Nov 7, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Devices/Host.h>
#ifdef HAVE_OPENMP
#include <omp.h>
#endif
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>

namespace TNL {
namespace Devices {   


bool Host::ompEnabled( true );
int Host::maxThreadsCount( -1 );

String Host::getDeviceType()
{
   return String( "Devices::Host" );
};

size_t Host::getFreeMemory()
{
   long pages = sysconf(_SC_PHYS_PAGES);
   long page_size = sysconf(_SC_PAGE_SIZE);
   return pages * page_size;
};

void Host::enableOMP()
{
   ompEnabled = true;
}

void Host::disableOMP()
{
   ompEnabled = false;
}

void Host::setMaxThreadsCount( int maxThreadsCount_ )
{
   maxThreadsCount = maxThreadsCount_;
#ifdef HAVE_OPENMP
   omp_set_num_threads( maxThreadsCount );
#endif
}

int Host::getMaxThreadsCount()
{
#ifdef HAVE_OPENMP
   if( maxThreadsCount == -1 )
      return omp_get_max_threads();
   return maxThreadsCount;
#else
   return 0;
#endif
}
 
int Host::getThreadIdx()
{
#ifdef HAVE_OPENMP
   return omp_get_thread_num();
#else
   return 0;
#endif
}

void Host::configSetup( Config::ConfigDescription& config, const String& prefix )
{
#ifdef HAVE_OPENMP
   config.addEntry< bool >( prefix + "omp-enabled", "Enable support of OpenMP.", true );
   config.addEntry<  int >( prefix + "omp-max-threads", "Set maximum number of OpenMP threads.", omp_get_max_threads() );
#else
   config.addEntry< bool >( prefix + "omp-enabled", "Enable support of OpenMP (not supported on this system).", false );
   config.addEntry<  int >( prefix + "omp-max-threads", "Set maximum number of OpenMP threads (not supported on this system).", 0 );
#endif
 
}
 
bool Host::setup( const Config::ParameterContainer& parameters,
                  const String& prefix )
{
   ompEnabled = parameters.getParameter< bool >( prefix + "omp-enabled" );
   maxThreadsCount = parameters.getParameter< int >( prefix + "omp-max-threads" );
   return true;
}

} // namespace Devices
} // namespace TNL

