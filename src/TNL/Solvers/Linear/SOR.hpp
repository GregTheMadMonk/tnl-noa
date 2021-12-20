/***************************************************************************
                          SOR.hpp  -  description
                             -------------------
    begin                : Nov 25, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include "SOR.h"
#include <TNL/Solvers/Linear/Utils/LinearResidueGetter.h>

namespace TNL {
namespace Solvers {
namespace Linear {

template< typename Matrix >
void
SOR< Matrix >::
configSetup( Config::ConfigDescription& config,
             const String& prefix )
{
   LinearSolver< Matrix >::configSetup( config, prefix );
   config.addEntry< double >( prefix + "sor-omega", "Relaxation parameter of the SOR method.", 1.0 );
}

template< typename Matrix >
bool
SOR< Matrix >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   if( parameters.checkParameter( prefix + "sor-omega" ) )
      this->setOmega( parameters.getParameter< double >( prefix + "sor-omega" ) );
   if( this->omega <= 0.0 || this->omega > 2.0 )
   {
      std::cerr << "Warning: The SOR method parameter omega is out of interval (0,2). The value is " << this->omega << " the method will not converge." << std::endl;
   }
   return LinearSolver< Matrix >::setup( parameters, prefix );
}

template< typename Matrix >
void SOR< Matrix > :: setOmega( const RealType& omega )
{
   this->omega = omega;
}

template< typename Matrix >
const typename SOR< Matrix > :: RealType& SOR< Matrix > :: getOmega( ) const
{
   return this->omega;
}

template< typename Matrix >
bool SOR< Matrix > :: solve( ConstVectorViewType b, VectorViewType x )
{
   const IndexType size = this->matrix->getRows();

   this->resetIterations();
   this->setResidue( this->getConvergenceResidue() + 1.0 );

   RealType bNorm = lpNorm( b, 2.0 );

   while( this->nextIteration() )
   {
      for( IndexType row = 0; row < size; row ++ )
         this->matrix->performSORIteration( b, row, x, this->getOmega() );
      this->setResidue( LinearResidueGetter::getResidue( *this->matrix, x, b, bNorm ) );
      this->refreshSolverMonitor();
   }
   this->setResidue( LinearResidueGetter::getResidue( *this->matrix, x, b, bNorm ) );
   this->refreshSolverMonitor();
   return this->checkConvergence();
};

} // namespace Linear
} // namespace Solvers
} // namespace TNL
