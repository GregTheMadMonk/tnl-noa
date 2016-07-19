/***************************************************************************
                          tnlSolverInitiator_impl.h  -  description
                             -------------------
    begin                : Feb 23, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#include <TNL/config/tnlParameterContainer.h>
#include <TNL/solvers/tnlMeshTypeResolver.h>
#include <TNL/solvers/tnlBuildConfigTags.h>
#include <TNL/solvers/linear/stationary/tnlSORSolver.h>
#include <TNL/solvers/linear/krylov/tnlCGSolver.h>
#include <TNL/solvers/linear/krylov/tnlBICGStabSolver.h>
#include <TNL/solvers/linear/krylov/tnlGMRESSolver.h>
#include <TNL/core/tnlHost.h>
#include <TNL/core/tnlCuda.h>

namespace TNL {

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename ConfigTag,
          bool enabled = tnlConfigTagReal< ConfigTag, Real >::enabled >
class tnlSolverInitiatorRealResolver{};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename ConfigTag,
          bool enabled = tnlConfigTagDevice< ConfigTag, Device >::enabled >
class tnlSolverInitiatorDeviceResolver{};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag,
          bool enabled = tnlConfigTagIndex< ConfigTag, Index >::enabled >
class tnlSolverInitiatorIndexResolver{};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename ConfigTag  >
bool tnlSolverInitiator< ProblemSetter, ConfigTag > :: run( const tnlParameterContainer& parameters )
{
   const tnlString& realType = parameters. getParameter< tnlString >( "real-type" );
   if( parameters. getParameter< int >( "verbose" ) )
     std::cout << "Setting RealType to   ... " << realType << std::endl;
   if( realType == "float" )
      return tnlSolverInitiatorRealResolver< ProblemSetter, float, ConfigTag >::run( parameters );
   if( realType == "double" )
      return tnlSolverInitiatorRealResolver< ProblemSetter, double, ConfigTag >::run( parameters );
   if( realType == "long-double" )
      return tnlSolverInitiatorRealResolver< ProblemSetter, long double, ConfigTag >::run( parameters );
   std::cerr << "The real type '" << realType << "' is not defined. " << std::endl;
   return false;
};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename ConfigTag >
class tnlSolverInitiatorRealResolver< ProblemSetter, Real, ConfigTag, true >
{
   public:
      static bool run( const tnlParameterContainer& parameters )
      {
         const tnlString& device = parameters. getParameter< tnlString >( "device" );
         if( parameters. getParameter< int >( "verbose" ) )
           std::cout << "Setting DeviceType to ... " << device << std::endl;

         if( device == "host" )
            return tnlSolverInitiatorDeviceResolver< ProblemSetter, Real, tnlHost, ConfigTag >::run( parameters );
         if( device == "cuda" )
            return tnlSolverInitiatorDeviceResolver< ProblemSetter, Real, tnlCuda, ConfigTag >::run( parameters );
         std::cerr << "The device '" << device << "' is not defined. " << std::endl;
         return false;
      }
};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename ConfigTag >
class tnlSolverInitiatorRealResolver< ProblemSetter, Real, ConfigTag, false >
{
   public:
      static bool run( const tnlParameterContainer& parameters )
      {
         std::cerr << "The real type " << parameters.getParameter< tnlString >( "real-type" ) << " is not supported." << std::endl;
         return false;
      }
};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename ConfigTag >
class tnlSolverInitiatorDeviceResolver< ProblemSetter, Real, Device, ConfigTag, true >
{
   public:
      static bool run( const tnlParameterContainer& parameters )
      {
         const tnlString& indexType = parameters. getParameter< tnlString >( "index-type" );
         if( parameters. getParameter< int >( "verbose" ) )
           std::cout << "Setting IndexType to  ... " << indexType << std::endl;
         if( indexType == "short-int" )
            return tnlSolverInitiatorIndexResolver< ProblemSetter, Real, Device, short int, ConfigTag >::run( parameters );
         if( indexType == "int" )
            return tnlSolverInitiatorIndexResolver< ProblemSetter, Real, Device, int, ConfigTag >::run( parameters );
         if( indexType == "long int" )
            return tnlSolverInitiatorIndexResolver< ProblemSetter, Real, Device, long int, ConfigTag >::run( parameters );
         std::cerr << "The index type '" << indexType << "' is not defined. " << std::endl;
         return false;
      }
};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename ConfigTag >
class tnlSolverInitiatorDeviceResolver< ProblemSetter, Real, Device, ConfigTag, false >
{
   public:
      static bool run( const tnlParameterContainer& parameters )
      {
         std::cerr << "The device " << parameters.getParameter< tnlString >( "device" ) << " is not supported." << std::endl;
         return false;
      }
};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag >
class tnlSolverInitiatorIndexResolver< ProblemSetter, Real, Device, Index, ConfigTag, false >
{
   public:
      static bool run( const tnlParameterContainer& parameters )
      {
         std::cerr << "The index " << parameters.getParameter< tnlString >( "index-type" ) << " is not supported." << std::endl;
         return false;
      }
};

template< template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter > class ProblemSetter,
          typename Real,
          typename Device,
          typename Index,
          typename ConfigTag >
class tnlSolverInitiatorIndexResolver< ProblemSetter, Real, Device, Index, ConfigTag, true >
{
   public:
      static bool run( const tnlParameterContainer& parameters )
      {
         return tnlMeshTypeResolver< ProblemSetter, Real, Device, Index, ConfigTag >::run( parameters );
      }
};

} // namespace TNL

