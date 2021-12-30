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

#include <TNL/Solvers/Linear/Jacobi.h>
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

/**
 * \brief Function returning available linear solvers.
 *
 * \return container with names of available linear solvers.
 */
inline std::vector<std::string>
getLinearSolverOptions()
{
   return {
      "jacobi",
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

/**
 * \brief Function returning available linear preconditioners.
 *
 * \return container with names of available linear preconditioners.
 */
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

/**
 * \brief Function returning shared pointer with linear solver given by its name in a form of a string.
 *
 * \tparam MatrixType is a type of matrix defining the system of linear equations.
 * \param name of the linear solver. The name can be one of the following:
 *    1. `jacobi`    - for the Jacobi solver - \ref TNL::Solvers::Linear::Jacobi.
 *    2. `sor`       - for SOR solver        - \ref TNL::Solvers::Linear::SOR.
 *    3. `cg`        - for CG solver         - \ref TNL::Solvers::Linear::CG.
 *    4. `bicgstab`  - for BICGStab solver   - \ref TNL::Solvers::Linear::BICGStab.
 *    5. `bicgstabl` - for BICGStabL solver  - \ref TNL::Solvers::Linear::BICGStabL.
 *    6. `gmres`     - for GMRES solver      - \ref TNL::Solvers::Linear::GMRES.
 *    7. `tfqmr`     - for TFQMR solver      - \ref TNL::Solvers::Linear::TFQMR.
 * \return shared pointer with given linear solver.
 *
 * The following example shows how to use this function:
 *
 * \includelineno Solvers/Linear/IterativeLinearSolverWithRuntimeTypesExample.cpp
 *
 * The result looks as follows:
 *
 * \include IterativeLinearSolverWithRuntimeTypesExample.out
 */
template< typename MatrixType >
std::shared_ptr< Linear::LinearSolver< MatrixType > >
getLinearSolver( std::string name )
{
   if( name == "jacobi" )
      return std::make_shared< Linear::Jacobi< MatrixType > >();
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

/**
 * \brief Function returning shared pointer with linear preconditioner given by its name in a form of a string.
 *
 * \tparam MatrixType is a type of matrix defining the system of linear equations.
 * \param name of the linear preconditioner. The name can be one of the following:
 *    1. `none`     - for none preconditioner
 *    2. `diagonal` - for diagonal/Jacobi preconditioner         - \ref TNL::Solvers::Linear::Preconditioners::Diagonal
 *    3. `ilu0`     - for ILU(0) preconditioner                  - \ref TNL::Solvers::Linear::Preconditioners::ILU0
 *    4. `ilut`     - for ILU with thresholding preconditioner   - \ref TNL::Solvers::Linear::Preconditioners::ILUT
 * \return shared pointer with given linear preconditioner.
 *
 * \includelineno Solvers/Linear/IterativeLinearSolverWithRuntimeTypesExample.cpp
 *
 * The result looks as follows:
 *
 * \include IterativeLinearSolverWithRuntimeTypesExample.out
 */
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
