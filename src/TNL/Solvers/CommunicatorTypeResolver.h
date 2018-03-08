/***************************************************************************
                          CommunicatorTypeResolver.h  
                             -------------------
    begin                : Feb 12, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Config/ParameterContainer.h>

#include <TNL/Solvers/SolverStarter.h>

#include <TNL/Communicators/NoDistrCommunicator.h>
#include <TNL/Communicators/MpiCommunicator.h>

namespace TNL {
namespace Solvers { 

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter, typename CommunicatorType > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename MeshType,
          typename ConfigTag >
class CommunicatorTypeResolver
{
   public:
      static bool run( const Config::ParameterContainer& parameters )
      {
#ifdef HAVE_MPI
         if(Communicators::MpiCommunicator::isDistributed())
         {     
               bool ret=ProblemSetter< Real, Device, Index, MeshType, ConfigTag, SolverStarter< ConfigTag >, Communicators::MpiCommunicator >::run( parameters );
               Communicators::MpiCommunicator::Finalize();      
               return ret;
         }
         Communicators::MpiCommunicator::Finalize();
#endif
         return ProblemSetter< Real, Device, Index, MeshType, ConfigTag, SolverStarter< ConfigTag >, Communicators::NoDistrCommunicator >::run( parameters );
         
      }
};

} // namespace Solvers
} // namespace TNL
