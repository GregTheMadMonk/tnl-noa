/***************************************************************************
                          LinearSolverTypeResolver.h  -  description
                             -------------------
    begin                : Sep 4, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovský

#pragma once

#include <vector>
#include <string>
#include <memory>

#include <TNL/Solvers/Linear/SOR.h>
#include <TNL/Solvers/Linear/CG.h>
#include <TNL/Solvers/Linear/BICGStab.h>
#include <TNL/Solvers/Linear/BICGStabL.h>
#include <TNL/Solvers/Linear/GMRES.h>
#include <TNL/Solvers/Linear/TFQMR.h>
#include <TNL/Solvers/Linear/UmfpackWrapper.h>
#include <TNL/Solvers/Linear/Preconditioners/Diagonal.h>
#include <TNL/Solvers/Linear/Preconditioners/ILU0.h>
#include <TNL/Solvers/Linear/Preconditioners/ILUT.h>

namespace TNL {
namespace Solvers {

inline std::vector<std::string>
getLinearSolverOptions()
{
   return {
      "sor",
      "cg",
      "bicgstab",
      "bicgstabl",
      "gmres",
      "tfqmr"
#ifdef HAVE_UMFPACK
      , "umfpack"
#endif
   };
}

inline std::vector<std::string>
getPreconditionerOptions()
{
   return {
      "none",
      "diagonal",
      "ilu0",
      "ilut"
   };
}

template< typename MatrixType >
std::shared_ptr< Linear::LinearSolver< MatrixType > >
getLinearSolver( std::string name )
{
   if( name == "sor" )
      return std::make_shared< Linear::SOR< MatrixType > >();
   if( name == "cg" )
      return std::make_shared< Linear::CG< MatrixType > >();
   if( name == "bicgstab" )
      return std::make_shared< Linear::BICGStab< MatrixType > >();
   if( name == "bicgstabl" )
      return std::make_shared< Linear::BICGStabL< MatrixType > >();
   if( name == "gmres" )
      return std::make_shared< Linear::GMRES< MatrixType > >();
   if( name == "tfqmr" )
      return std::make_shared< Linear::TFQMR< MatrixType > >();
#ifdef HAVE_UMFPACK
   if( discreteSolver == "umfpack" )
      return std::make_shared< Linear::UmfpackWrapper< MatrixType > >();
#endif

   std::string options;
   for( auto o : getLinearSolverOptions() )
      if( options.empty() )
         options += o;
      else
         options += ", " + o;

   std::cerr << "Unknown semi-implicit discrete solver " << name << ". It can be only: " << options << "." << std::endl;

   return nullptr;
}

template< typename MatrixType >
std::shared_ptr< Linear::Preconditioners::Preconditioner< MatrixType > >
getPreconditioner( std::string name )
{

   if( name == "none" )
      return nullptr;
   if( name == "diagonal" )
      return std::make_shared< Linear::Preconditioners::Diagonal< MatrixType > >();
   if( name == "ilu0" )
      return std::make_shared< Linear::Preconditioners::ILU0< MatrixType > >();
   if( name == "ilut" )
      return std::make_shared< Linear::Preconditioners::ILUT< MatrixType > >();

   std::string options;
   for( auto o : getPreconditionerOptions() )
      if( options.empty() )
         options += o;
      else
         options += ", " + o;

   std::cerr << "Unknown preconditioner " << name << ". It can be only: " << options << "." << std::endl;

   return nullptr;
}

} // namespace Solvers
} // namespace TNL
