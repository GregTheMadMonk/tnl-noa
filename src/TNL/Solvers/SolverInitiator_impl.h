/***************************************************************************
                          SolverInitiator_impl.h  -  description
                             -------------------
    begin                : Feb 23, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Solvers/SolverInitiator.h>

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Meshes/TypeResolver/TypeResolver.h>
#include <TNL/Solvers/BuildConfigTags.h>
#include <TNL/Solvers/SolverStarter.h>
#include <TNL/Meshes/DummyMesh.h>

#include <TNL/Communicators/NoDistrCommunicator.h>
#include <TNL/Communicators/MpiCommunicator.h>

namespace TNL {
namespace Solvers {   

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter, typename CommunicatorType > class ProblemSetter,
          typename Real,
          typename ConfigTag,
          bool enabled = ConfigTagReal< ConfigTag, Real >::enabled >
class SolverInitiatorRealResolver {};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter, typename CommunicatorType > class ProblemSetter,
          typename Real,
          typename Device,
          typename ConfigTag,
          bool enabled = ConfigTagDevice< ConfigTag, Device >::enabled >
class SolverInitiatorDeviceResolver {};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter, typename CommunicatorType > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag,
          bool enabled = ConfigTagIndex< ConfigTag, Index >::enabled >
class SolverInitiatorIndexResolver {};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter, typename CommunicatorType > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag,
          bool enabled = true  >
class CommunicatorTypeResolver {};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter, typename CommunicatorType > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag,
          typename CommunicatorType,
          bool enabled = ConfigTagMeshResolve< ConfigTag >::enabled >
class SolverInitiatorMeshResolver {};


template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter, typename CommunicatorType > class ProblemSetter,
          typename ConfigTag  >
bool SolverInitiator< ProblemSetter, ConfigTag > :: run( const Config::ParameterContainer& parameters )
{
   const String& realType = parameters. getParameter< String >( "real-type" );
   if( realType == "float" )
      return SolverInitiatorRealResolver< ProblemSetter, float, ConfigTag >::run( parameters );
   if( realType == "double" )
      return SolverInitiatorRealResolver< ProblemSetter, double, ConfigTag >::run( parameters );
   if( realType == "long-double" )
      return SolverInitiatorRealResolver< ProblemSetter, long double, ConfigTag >::run( parameters );
   std::cerr << "The real type '" << realType << "' is not defined. " << std::endl;
   return false;
};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter, typename CommunicatorType > class ProblemSetter,
          typename Real,
          typename ConfigTag >
class SolverInitiatorRealResolver< ProblemSetter, Real, ConfigTag, true >
{
   public:
      static bool run( const Config::ParameterContainer& parameters )
      {
         const String& device = parameters. getParameter< String >( "device" );
         if( device == "host" )
            return SolverInitiatorDeviceResolver< ProblemSetter, Real, Devices::Host, ConfigTag >::run( parameters );
         if( device == "cuda" )
            return SolverInitiatorDeviceResolver< ProblemSetter, Real, Devices::Cuda, ConfigTag >::run( parameters );
         std::cerr << "The device '" << device << "' is not defined. " << std::endl;
         return false;
      }
};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter, typename CommunicatorType > class ProblemSetter,
          typename Real,
          typename ConfigTag >
class SolverInitiatorRealResolver< ProblemSetter, Real, ConfigTag, false >
{
   public:
      static bool run( const Config::ParameterContainer& parameters )
      {
         std::cerr << "The real type " << parameters.getParameter< String >( "real-type" ) << " is not supported." << std::endl;
         return false;
      }
};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter, typename CommunicatorType > class ProblemSetter,
          typename Real,
          typename Device,
          typename ConfigTag >
class SolverInitiatorDeviceResolver< ProblemSetter, Real, Device, ConfigTag, true >
{
   public:
      static bool run( const Config::ParameterContainer& parameters )
      {
         const String& indexType = parameters. getParameter< String >( "index-type" );
         if( indexType == "short-int" )
            return SolverInitiatorIndexResolver< ProblemSetter, Real, Device, short int, ConfigTag >::run( parameters );
         if( indexType == "int" )
            return SolverInitiatorIndexResolver< ProblemSetter, Real, Device, int, ConfigTag >::run( parameters );
         if( indexType == "long int" )
            return SolverInitiatorIndexResolver< ProblemSetter, Real, Device, long int, ConfigTag >::run( parameters );
         std::cerr << "The index type '" << indexType << "' is not defined. " << std::endl;
         return false;
      }
};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter, typename CommunicatorType > class ProblemSetter,
          typename Real,
          typename Device,
          typename ConfigTag >
class SolverInitiatorDeviceResolver< ProblemSetter, Real, Device, ConfigTag, false >
{
   public:
      static bool run( const Config::ParameterContainer& parameters )
      {
         std::cerr << "The device " << parameters.getParameter< String >( "device" ) << " is not supported." << std::endl;
         return false;
      }
};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter, typename CommunicatorType > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag >
class SolverInitiatorIndexResolver< ProblemSetter, Real, Device, Index, ConfigTag, false >
{
   public:
      static bool run( const Config::ParameterContainer& parameters )
      {
         std::cerr << "The index " << parameters.getParameter< String >( "index-type" ) << " is not supported." << std::endl;
         return false;
      }
};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter, typename CommunicatorType > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag >
class SolverInitiatorIndexResolver< ProblemSetter, Real, Device, Index, ConfigTag, true >
{
   public:
      static bool run( const Config::ParameterContainer& parameters )
      {
         return CommunicatorTypeResolver< ProblemSetter, Real, Device, Index, ConfigTag, true >::run( parameters );
      }
};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter, typename CommunicatorType > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag >
class CommunicatorTypeResolver< ProblemSetter, Real, Device, Index, ConfigTag, true >
{
   public:
      static bool run( const Config::ParameterContainer& parameters )
      {
         if( Communicators::MpiCommunicator::isDistributed() )
            return SolverInitiatorMeshResolver< ProblemSetter, Real, Device, Index, ConfigTag, Communicators::MpiCommunicator >::run( parameters );
         return SolverInitiatorMeshResolver< ProblemSetter, Real, Device, Index, ConfigTag, Communicators::NoDistrCommunicator >::run( parameters );
      }
};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter, typename CommunicatorType > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag,
          typename CommunicatorType >
class SolverInitiatorMeshResolver< ProblemSetter, Real, Device, Index, ConfigTag, CommunicatorType, false >
{
   public:
      static bool run( const Config::ParameterContainer& parameters )
      {
         return ProblemSetter< Real,
                               Device,
                               Index,
                               Meshes::DummyMesh< Real, Device, Index >,
                               ConfigTag,
                               SolverStarter< ConfigTag >, CommunicatorType >::template run< Real, Device, Index, ConfigTag >( parameters );
      }
};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter, typename CommunicatorType > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag,
          typename CommunicatorType >
class SolverInitiatorMeshResolver< ProblemSetter, Real, Device, Index, ConfigTag,CommunicatorType, true >
{
   // wrapper for MeshTypeResolver
   template< typename MeshType >
   using ProblemSetterWrapper = ProblemSetter< Real, Device, Index, MeshType, ConfigTag, SolverStarter< ConfigTag >, CommunicatorType >;

   public:
      static bool run( const Config::ParameterContainer& parameters )
      {
         const String& meshFileName = parameters.getParameter< String >( "mesh" );
         return Meshes::resolveMeshType< ConfigTag, Device, ProblemSetterWrapper >( meshFileName, parameters );
      }
};

} // namespace Solvers
} // namespace TNL
