#pragma once

#ifdef HAVE_UMFPACK

#include "tnlUmfpackWrapper.h"

template< typename Preconditioner >
tnlUmfpackWrapper< tnlCSRMatrix< double, tnlHost, int >, Preconditioner >::
tnlUmfpackWrapper()
{}

template< typename Preconditioner >
void
tnlUmfpackWrapper< tnlCSRMatrix< double, tnlHost, int >, Preconditioner >::
configSetup( tnlConfigDescription& config,
             const tnlString& prefix )
{
}

template< typename Preconditioner >
bool
tnlUmfpackWrapper< tnlCSRMatrix< double, tnlHost, int >, Preconditioner >::
setup( const tnlParameterContainer& parameters,
       const tnlString& prefix )
{
    return true;    
}

template< typename Preconditioner >
void tnlUmfpackWrapper< tnlCSRMatrix< double, tnlHost, int >, Preconditioner >::
setMatrix( const MatrixType& matrix )
{
    this -> matrix = &matrix;
}

template< typename Preconditioner >
void tnlUmfpackWrapper< tnlCSRMatrix< double, tnlHost, int >, Preconditioner >::
setPreconditioner( const Preconditioner& preconditioner )
{
    this -> preconditioner = &preconditioner;
}


template< typename Preconditioner >
    template< typename Vector, typename ResidueGetter >
bool tnlUmfpackWrapper< tnlCSRMatrix< double, tnlHost, int >, Preconditioner >::
solve( const Vector& b,
       Vector& x )
{
    tnlAssert( matrix->getRows() == matrix->getColumns(), );
    tnlAssert( matrix->getColumns() == x.getSize() && matrix->getColumns() == b.getSize(), );

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
//    Control[ UMFPACK_PRL ] = 2;

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
        cerr << "error: symbolic reordering failed" << endl;
        umfpack_di_report_status( Control, status );
//       umfpack_di_report_control( Control );
//       umfpack_di_report_info( Control, Info );
        goto finished;
    }

    // numeric factorization
    status = umfpack_di_numeric( matrix->rowPointers.getData(),
                                 matrix->columnIndexes.getData(),
                                 matrix->values.getData(),
                                 Symbolic, &Numeric, Control, Info );
    if( status != UMFPACK_OK ) {
        cerr << "error: numeric factorization failed" << endl;
        umfpack_di_report_status( Control, status );
//       umfpack_di_report_control( Control );
//       umfpack_di_report_info( Control, Info );
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
        cerr << "error: umfpack_di_solve failed" << endl;
            umfpack_di_report_status( Control, status );
//           umfpack_di_report_control( Control );
//           umfpack_di_report_info( Control, Info );
        goto finished;
    }

//    umfpack_di_report_info( Control, Info );

finished:
    if( Symbolic )
        umfpack_di_free_symbolic( &Symbolic );
    if( Numeric )
        umfpack_di_free_numeric( &Numeric );

    this->setResidue( ResidueGetter::getResidue( *matrix, x, b, bNorm ) );
    this->refreshSolverMonitor( true );
    return status == UMFPACK_OK;
};

#endif
