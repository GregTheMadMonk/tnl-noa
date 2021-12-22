/***************************************************************************
                          Jacobi.hpp  -  description
                             -------------------
    begin                : Dec 22, 2021
    copyright            : (C) 2021 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Solvers/Linear/LinearSolver.h>
#include <TNL/Solvers/Linear/Utils/LinearResidueGetter.h>

namespace TNL {
   namespace Solvers {
      namespace Linear {

template< typename Matrix >
void
Jacobi< Matrix >::
configSetup( Config::ConfigDescription& config, const String& prefix )
{
   LinearSolver< Matrix >::configSetup( config, prefix );
   config.addEntry< double >( prefix + "jacobi-omega", "Relaxation parameter of the weighted/damped Jacobi method.", 1.0 );
}

template< typename Matrix >
bool
Jacobi< Matrix >::
setup( const Config::ParameterContainer& parameters, const String& prefix ) override
{
   if( parameters.checkParameter( prefix + "jacobi-omega" ) )
      this->setOmega( parameters.getParameter< double >( prefix + "jacobi-omega" ) );
   if( this->omega <= 0.0 || this->omega > 2.0 )
   {
      std::cerr << "Warning: The Jacobi method parameter omega is out of interval (0,2). The value is " << this->omega << " the method will not converge." << std::endl;
   }
   return LinearSolver< Matrix >::setup( parameters, prefix );
}

template< typename Matrix >
void
Jacobi< Matrix >::
setOmega( RealType omega )
{
   this->omega = omega;
}

template< typename Matrix >
auto
Jacobi< Matrix >::
getOmega() const -> RealType
{
   return omega;
}

template< typename Matrix >
bool
Jacobi< Matrix >::
solve( ConstVectorViewType b, VectorViewType x ) override
{
   const IndexType size = this->matrix->getRows();

   Containers::Vector< RealType, DeviceType, IndexType > aux;
   aux.setSize( size );

   this->resetIterations();
   this->setResidue( this->getConvergenceResidue() + 1.0 );

   RealType bNorm = b.lpNorm( ( RealType ) 2.0 );
   aux = x;
   while( this->nextIteration() )
   {
      for( IndexType row = 0; row < size; row ++ )
         this->matrix->performJacobiIteration( b, row, x, aux, this->getOmega() );
      for( IndexType row = 0; row < size; row ++ )
         this->matrix->performJacobiIteration( b, row, aux, x, this->getOmega() );
      this->setResidue( LinearResidueGetter::getResidue( *this->matrix, x, b, bNorm ) );
   }
   this->setResidue( LinearResidueGetter::getResidue( *this->matrix, x, b, bNorm ) );
   return this->checkConvergence();
}

      } // namespace Linear
   } // namespace Solvers
} // namespace TNL
