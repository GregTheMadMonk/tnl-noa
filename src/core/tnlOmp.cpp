/***************************************************************************
                          tnlOmp.cpp  -  description
                             -------------------
    begin                : Mar 4, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
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

#include <core/tnlOmp.h>
#ifdef HAVE_OPENMP
#include <omp.h>
#endif

bool tnlOmp::enabled( true );
int tnlOmp::maxThreadsCount( -1 );

void tnlOmp::enable()
{
   enabled = true;
}

void tnlOmp::disable()
{
   enabled = false;
}

void tnlOmp::setMaxThreadsCount( int maxThreadsCount_ )
{
   maxThreadsCount = maxThreadsCount_;
#ifdef HAVE_OPENMP   
   omp_set_num_threads( maxThreadsCount );
#endif   
}

int tnlOmp::getMaxThreadsCount()
{
#ifdef HAVE_OPENMP
   if( maxThreadsCount == -1 )
      return omp_get_max_threads();
   return maxThreadsCount;
#else
   return 0;
#endif
}
      
int tnlOmp::getThreadIdx()
{
#ifdef HAVE_OPENMP
   return omp_get_thread_num();
#else
   return 0;
#endif  
}

void tnlOmp::configSetup( tnlConfigDescription& config, const tnlString& prefix )
{
#ifdef HAVE_OPENMP
   config.addEntry< bool >( prefix + "omp-enabled", "Enable support of OpenMP.", true );
   config.addEntry<  int >( prefix + "omp-max-threads", "Set maximum number of OpenMP threads.", omp_get_max_threads() );
#else
   config.addEntry< bool >( prefix + "omp-enabled", "Enable support of OpenMP (not supported on this system).", false );
   config.addEntry<  int >( prefix + "omp-max-threads", "Set maximum number of OpenMP threads (not supported on this system).", 0 );
#endif
   
}
      
bool tnlOmp::setup( const tnlParameterContainer& parameters,
                    const tnlString& prefix )
{
   enabled = parameters.getParameter< bool >( prefix + "omp-enabled" );
   maxThreadsCount = parameters.getParameter< int >( prefix + "omp-max-threads" );
   return true;
}

