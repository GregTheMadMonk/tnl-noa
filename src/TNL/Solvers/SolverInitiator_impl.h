/***************************************************************************
                          SolverInitiator_impl.h  -  description
                             -------------------
    begin                : Feb 23, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/Config/ParameterContainer.h>
#include <TNL/Solvers/MeshTypeResolver.h>
#include <TNL/Solvers/BuildConfigTags.h>
#include <TNL/Solvers/Linear/SOR.h>
#include <TNL/Solvers/Linear/CG.h>
#include <TNL/Solvers/Linear/BICGStab.h>
#include <TNL/Solvers/Linear/GMRES.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

namespace TNL {
namespace Solvers {   

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename ConfigTag,
          bool enabled = ConfigTagReal< ConfigTag, Real >::enabled >
class SolverInitiatorRealResolver{};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename ConfigTag,
          bool enabled = ConfigTagDevice< ConfigTag, Device >::enabled >
class SolverInitiatorDeviceResolver{};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag,
          bool enabled = ConfigTagIndex< ConfigTag, Index >::enabled >
class SolverInitiatorIndexResolver{};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename ConfigTag  >
bool SolverInitiator< ProblemSetter, ConfigTag > :: run( const Config::ParameterContainer& parameters )
{
   const String& realType = parameters. getParameter< String >( "real-type" );
   if( parameters. getParameter< int >( "verbose" ) )
     std::cout << "Setting RealType to   ... " << realType << std::endl;
   if( realType == "float" )
      return SolverInitiatorRealResolver< ProblemSetter, float, ConfigTag >::run( parameters );
   if( realType == "double" )
      return SolverInitiatorRealResolver< ProblemSetter, double, ConfigTag >::run( parameters );
   if( realType == "long-double" )
      return SolverInitiatorRealResolver< ProblemSetter, long double, ConfigTag >::run( parameters );
   std::cerr << "The real type '" << realType << "' is not defined. " << std::endl;
   return false;
};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename ConfigTag >
class SolverInitiatorRealResolver< ProblemSetter, Real, ConfigTag, true >
{
   public:
      static bool run( const Config::ParameterContainer& parameters )
      {
         const String& device = parameters. getParameter< String >( "device" );
         if( parameters. getParameter< int >( "verbose" ) )
           std::cout << "Setting DeviceType to ... " << device << std::endl;

         if( device == "host" )
            return SolverInitiatorDeviceResolver< ProblemSetter, Real, Devices::Host, ConfigTag >::run( parameters );
         if( device == "cuda" )
            return SolverInitiatorDeviceResolver< ProblemSetter, Real, Devices::Cuda, ConfigTag >::run( parameters );
         std::cerr << "The device '" << device << "' is not defined. " << std::endl;
         return false;
      }
};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
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

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename ConfigTag >
class SolverInitiatorDeviceResolver< ProblemSetter, Real, Device, ConfigTag, true >
{
   public:
      static bool run( const Config::ParameterContainer& parameters )
      {
         const String& indexType = parameters. getParameter< String >( "index-type" );
         if( parameters. getParameter< int >( "verbose" ) )
           std::cout << "Setting IndexType to  ... " << indexType << std::endl;
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

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
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

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
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

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag >
class SolverInitiatorIndexResolver< ProblemSetter, Real, Device, Index, ConfigTag, true >
{
   public:
      static bool run( const Config::ParameterContainer& parameters )
      {
         return MeshTypeResolver< ProblemSetter, Real, Device, Index, ConfigTag >::run( parameters );
      }
};

} // namespace Solvers
} // namespace TNL

