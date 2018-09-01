/***************************************************************************
                          Host.cpp  -  description
                             -------------------
    begin                : Nov 7, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

#include <TNL/Devices/Host.h>
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
   config.addEntry< bool >( prefix + "openmp-enabled", "Enable support of OpenMP.", true );
   config.addEntry<  int >( prefix + "openmp-max-threads", "Set maximum number of OpenMP threads.", omp_get_max_threads() );
#else
   config.addEntry< bool >( prefix + "openmp-enabled", "Enable support of OpenMP (not supported on this system).", false );
   config.addEntry<  int >( prefix + "openmp-max-threads", "Set maximum number of OpenMP threads (not supported on this system).", 0 );
#endif
 
}
 
bool Host::setup( const Config::ParameterContainer& parameters,
                  const String& prefix )
{
   if( parameters.getParameter< bool >( prefix + "openmp-enabled" ) ) {
#ifdef HAVE_OPENMP
      enableOMP();
#else
      std::cerr << "OpenMP is not supported - please recompile the TNL library with OpenMP." << std::endl;
      return false;
#endif
   }
   else
      disableOMP();
   const int threadsCount = parameters.getParameter< int >( prefix + "openmp-max-threads" );
   if( threadsCount > 1 && ! isOMPEnabled() )
      std::cerr << "Warning: openmp-max-threads was set to " << threadsCount << ", but OpenMP is disabled." << std::endl;
   setMaxThreadsCount( threadsCount );
   return true;
}

} // namespace Devices
} // namespace TNL
