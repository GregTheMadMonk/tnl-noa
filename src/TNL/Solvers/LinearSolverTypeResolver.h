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

template< typename MatrixType >
std::shared_ptr< Linear::LinearSolver< MatrixType > >
getLinearSolver( const Config::ParameterContainer& parameters )
{
   const String& discreteSolver = parameters.getParameter< String >( "discrete-solver" );

   if( discreteSolver == "sor" )
      return std::make_shared< Linear::SOR< MatrixType > >();
   if( discreteSolver == "cg" )
      return std::make_shared< Linear::CG< MatrixType > >();
   if( discreteSolver == "bicgstab" )
      return std::make_shared< Linear::BICGStab< MatrixType > >();
   if( discreteSolver == "bicgstabl" )
      return std::make_shared< Linear::BICGStabL< MatrixType > >();
   if( discreteSolver == "gmres" )
      return std::make_shared< Linear::GMRES< MatrixType > >();
   if( discreteSolver == "tfqmr" )
      return std::make_shared< Linear::TFQMR< MatrixType > >();
#ifdef HAVE_UMFPACK
   if( discreteSolver == "umfpack" )
      return std::make_shared< Linear::UmfpackWrapper< MatrixType > >();
#endif

   std::cerr << "Unknown semi-implicit discrete solver " << discreteSolver << ". It can be only: sor, cg, bicgstab, bicgstabl, gmres, tfqmr";
#ifdef HAVE_UMFPACK
   std::cerr << ", umfpack"
#endif
   std::cerr << "." << std::endl;

   return nullptr;
}

template< typename MatrixType >
std::shared_ptr< Linear::Preconditioners::Preconditioner< MatrixType > >
getPreconditioner( const Config::ParameterContainer& parameters )
{
   const String& preconditioner = parameters.getParameter< String >( "preconditioner" );

   if( preconditioner == "none" )
      return nullptr;
   if( preconditioner == "diagonal" )
      return std::make_shared< Linear::Preconditioners::Diagonal< MatrixType > >();
   if( preconditioner == "ilu0" )
      return std::make_shared< Linear::Preconditioners::ILU0< MatrixType > >();
   if( preconditioner == "ilut" )
      return std::make_shared< Linear::Preconditioners::ILUT< MatrixType > >();

   std::cerr << "Unknown preconditioner " << preconditioner << ". It can be only: none, diagonal, ilu0, ilut." << std::endl;
   return nullptr;
}

} // namespace Solvers
} // namespace TNL
