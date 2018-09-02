/***************************************************************************
                          SOR_impl.h  -  description
                             -------------------
    begin                : Nov 25, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Solvers/Linear/SOR.h>
#include <TNL/Solvers/Linear/LinearResidueGetter.h>

namespace TNL {
namespace Solvers {
namespace Linear {   

template< typename Matrix, typename Preconditioner >
SOR< Matrix, Preconditioner > :: SOR()
: omega( 1.0 )
{
   /****
    * Clearing the shared pointer means that there is no
    * preconditioner set.
    */
   this->preconditioner.clear();   
}

template< typename Matrix, typename Preconditioner >
String SOR< Matrix, Preconditioner > :: getType() const
{
   return String( "SOR< " ) +
          this->matrix -> getType() + ", " +
          this->preconditioner -> getType() + " >";
}

template< typename Matrix,
          typename Preconditioner >
void
SOR< Matrix, Preconditioner >::
configSetup( Config::ConfigDescription& config,
             const String& prefix )
{
   //IterativeSolver< RealType, IndexType >::configSetup( config, prefix );
   config.addEntry< double >( prefix + "sor-omega", "Relaxation parameter of the SOR method.", 1.0 );
}

template< typename Matrix,
          typename Preconditioner >
bool
SOR< Matrix, Preconditioner >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   IterativeSolver< RealType, IndexType >::setup( parameters, prefix );
   this->setOmega( parameters.getParameter< double >( prefix + "sor-omega" ) );
   if( this->omega <= 0.0 || this->omega > 2.0 )
   {
      std::cerr << "Warning: The SOR method parameter omega is out of interval (0,2). The value is " << this->omega << " the method will not converge." << std::endl;
   }
   return true;
}

template< typename Matrix, typename Preconditioner >
void SOR< Matrix, Preconditioner > :: setOmega( const RealType& omega )
{
   this->omega = omega;
}

template< typename Matrix, typename Preconditioner >
const typename SOR< Matrix, Preconditioner > :: RealType& SOR< Matrix, Preconditioner > :: getOmega( ) const
{
   return this->omega;
}

template< typename Matrix, typename Preconditioner >
bool SOR< Matrix, Preconditioner > :: solve( ConstVectorViewType b, VectorViewType x )
{
   const IndexType size = this->matrix->getRows();

   this->resetIterations();
   this->setResidue( this->getConvergenceResidue() + 1.0 );

   RealType bNorm = b.lpNorm( ( RealType ) 2.0 );

   while( this->nextIteration() )
   {
      for( IndexType row = 0; row < size; row ++ )
         this->matrix->performSORIteration( b, row, x, this->getOmega() );
      // FIXME: the LinearResidueGetter works only on the host
      this->setResidue( LinearResidueGetter::getResidue( *this->matrix, x, b, bNorm ) );
      this->refreshSolverMonitor();
   }
   this->setResidue( LinearResidueGetter::getResidue( *this->matrix, x, b, bNorm ) );
   this->refreshSolverMonitor( true );
   return this->checkConvergence();
};

} // namespace Linear
} // namespace Solvers
} // namespace TNL
