

#pragma once

#ifdef HAVE_UMFPACK

#include "UmfpackWrapper.h"

namespace TNL {
namespace Solvers {
namespace Linear {   

template< typename Preconditioner >
UmfpackWrapper< CSR< double, Devices::Host, int >, Preconditioner >::
UmfpackWrapper()
{}

template< typename Preconditioner >
void
UmfpackWrapper< CSR< double, Devices::Host, int >, Preconditioner >::
configSetup( Config::ConfigDescription& config,
             const String& prefix )
{
}

template< typename Preconditioner >
bool
UmfpackWrapper< CSR< double, Devices::Host, int >, Preconditioner >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
    return true;    
}

template< typename Preconditioner >
void UmfpackWrapper< CSR< double, Devices::Host, int >, Preconditioner >::
setMatrix( const MatrixType& matrix )
{
    this -> matrix = &matrix;
}

template< typename Preconditioner >
void UmfpackWrapper< CSR< double, Devices::Host, int >, Preconditioner >::
setPreconditioner( const Preconditioner& preconditioner )
{
    this -> preconditioner = &preconditioner;
}


template< typename Preconditioner >
    template< typename Vector, typename ResidueGetter >
bool UmfpackWrapper< CSR< double, Devices::Host, int >, Preconditioner >::
solve( const Vector& b,
       Vector& x )
{
    Assert( matrix->getRows() == matrix->getColumns(), );
    Assert( matrix->getColumns() == x.getSize() && matrix->getColumns() == b.getSize(), );

    const IndexType size = matrix -> getRows();

    this->resetIterations();
    this->setResidue( this -> getConvergenceResidue() + 1.0 );

    RealType bNorm = b. lpNorm( ( RealType ) 2.0 );

    // UMFPACK objects
    void* Symbolic = nullptr;
    void* Numeric = nullptr;

    int status = UMFPACK_OK;
    double Control[ UMFPACK_CONTROL ];
    double Info[ UMFPACK_INFO ];

    // umfpack expects Compressed Sparse Column format, we have Compressed Sparse Row
    // so we need to solve  A^T * x = rhs
    int system_type = UMFPACK_Aat;

    // symbolic reordering of the sparse matrix
    status = umfpack_di_symbolic( size, size,
                                  matrix->rowPointers.getData(),
                                  matrix->columnIndexes.getData(),
                                  matrix->values.getData(),
                                  &Symbolic, Control, Info );
    if( status != UMFPACK_OK ) {
        std::cerr << "error: symbolic reordering failed" << std::endl;
        goto finished;
    }

    // numeric factorization
    status = umfpack_di_numeric( matrix->rowPointers.getData(),
                                 matrix->columnIndexes.getData(),
                                 matrix->values.getData(),
                                 Symbolic, &Numeric, Control, Info );
    if( status != UMFPACK_OK ) {
        std::cerr << "error: numeric factorization failed" << std::endl;
        goto finished;
    }

    // solve with specified right-hand-side
    status = umfpack_di_solve( system_type,
                               matrix->rowPointers.getData(),
                               matrix->columnIndexes.getData(),
                               matrix->values.getData(),
                               x.getData(),
                               b.getData(),
                               Numeric, Control, Info );
    if( status != UMFPACK_OK ) {
        std::cerr << "error: umfpack_di_solve failed" << std::endl;
        goto finished;
    }

finished:
    if( status != UMFPACK_OK ) {
        // increase print level for reports
        Control[ UMFPACK_PRL ] = 2;
        umfpack_di_report_status( Control, status );
//        umfpack_di_report_control( Control );
//        umfpack_di_report_info( Control, Info );
    }

    if( Symbolic )
        umfpack_di_free_symbolic( &Symbolic );
    if( Numeric )
        umfpack_di_free_numeric( &Numeric );

    this->setResidue( ResidueGetter::getResidue( *matrix, x, b, bNorm ) );
    this->refreshSolverMonitor( true );
    return status == UMFPACK_OK;
};

} // namespace Linear
} // namespace Solvers
} // namespace TNL

#endif
