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
#include <TNL/Functional.h>

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
   config.addEntry< int >( prefix + "residue-period", "Says after how many iterations the reside is recomputed.", 4 );
}

template< typename Matrix >
bool
Jacobi< Matrix >::
setup( const Config::ParameterContainer& parameters, const String& prefix )
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
solve( ConstVectorViewType b, VectorViewType x )
{
   const IndexType size = this->matrix->getRows();
   Containers::Vector< RealType, DeviceType, IndexType > aux;
   aux.setSize( size );

   /////
   // Fetch diagonal elements
   this->diagonal.setSize( size );
   auto diagonalView = this->diagonal.getView();
   auto fetch_diagonal = [=] __cuda_callable__ ( IndexType rowIdx, IndexType localIdx, const IndexType& columnIdx, const RealType& value ) mutable {
      if( columnIdx == rowIdx ) diagonalView[ rowIdx ] = value;
   };
   this->matrix->forAllElements( fetch_diagonal );

   this->resetIterations();
   this->setResidue( this->getConvergenceResidue() + 1.0 );

   auto bView = b.getView();
   auto xView = x.getView();
   auto auxView = aux.getView();
   RealType bNorm = lpNorm( b, ( RealType ) 2.0 );
   aux = x;
   while( this->nextIteration() )
   {
      this->performIteration( bView, diagonalView, xView, auxView );
      if( this->getIterations() % this->residuePeriod == 0 )
         this->setResidue( LinearResidueGetter::getResidue( *this->matrix, x, b, bNorm ) );
      this->currentIteration++;
      this->performIteration( bView, diagonalView, auxView, xView );
      if( ( this->getIterations() ) % this->residuePeriod == 0 )
         this->setResidue( LinearResidueGetter::getResidue( *this->matrix, x, b, bNorm ) );

   }
   this->setResidue( LinearResidueGetter::getResidue( *this->matrix, x, b, bNorm ) );
   return this->checkConvergence();
}

template< typename Matrix >
void
Jacobi< Matrix >::
performIteration( const ConstVectorViewType& b,
                  const ConstVectorViewType& diagonalView,
                  const ConstVectorViewType& in,
                  VectorViewType& out ) const
{
   const RealType omega_ = this->omega;
   auto fetch = [=] __cuda_callable__ ( IndexType rowIdx, IndexType columnIdx, const RealType& value ) {
         return value * in[ columnIdx ];
   };
   auto keep = [=] __cuda_callable__ ( IndexType rowIdx, const RealType& value ) mutable {
      out[ rowIdx ] = in[ rowIdx ] + omega_ / diagonalView[ rowIdx ] * ( b[ rowIdx ] - value );
   };
   this->matrix->reduceAllRows( fetch, TNL::Plus{}, keep, 0.0 );
}

      } // namespace Linear
   } // namespace Solvers
} // namespace TNL
