/***************************************************************************
                          Jacobi.h  -  description
                             -------------------
    begin                : Jul 30, 2007
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

#pragma once

#include "LinearSolver.h"

#include <TNL/Containers/Vector.h>
#include <TNL/Solvers/Linear/Utils/LinearResidueGetter.h>

namespace TNL {
namespace Solvers {
namespace Linear {

template< typename Matrix >
class Jacobi
: public LinearSolver< Matrix >
{
   using Base = LinearSolver< Matrix >;
public:
   using RealType = typename Base::RealType;
   using DeviceType = typename Base::DeviceType;
   using IndexType = typename Base::IndexType;
   using VectorViewType = typename Base::VectorViewType;
   using ConstVectorViewType = typename Base::ConstVectorViewType;

   static void configSetup( Config::ConfigDescription& config,
                            const String& prefix = "" )
   {
      LinearSolver< Matrix >::configSetup( config, prefix );
      config.addEntry< double >( prefix + "jacobi-omega", "Relaxation parameter of the weighted/damped Jacobi method.", 1.0 );
   }

   bool setup( const Config::ParameterContainer& parameters,
               const String& prefix = "" ) override
   {
      if( parameters.checkParameter( prefix + "jacobi-omega" ) )
         this->setOmega( parameters.getParameter< double >( prefix + "jacobi-omega" ) );
      if( this->omega <= 0.0 || this->omega > 2.0 )
      {
         std::cerr << "Warning: The Jacobi method parameter omega is out of interval (0,2). The value is " << this->omega << " the method will not converge." << std::endl;
      }
      return LinearSolver< Matrix >::setup( parameters, prefix );
   }

   void setOmega( RealType omega )
   {
      this->omega = omega;
   }

   RealType getOmega() const
   {
      return omega;
   }

   bool solve( ConstVectorViewType b, VectorViewType x ) override
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
      this->refreshSolverMonitor();
      return this->checkConvergence();
   }

protected:
   RealType omega = 0.0;
};

} // namespace Linear
} // namespace Solvers
} // namespace TNL
