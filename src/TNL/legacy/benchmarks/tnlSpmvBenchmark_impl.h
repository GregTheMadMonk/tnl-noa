/***************************************************************************
                          tnlSpmvBenchmark_impl.h  -  description
                             -------------------
    begin                : Dec 29, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLSPMVBENCHMARK_IMPL_H_
#define TNLSPMVBENCHMARK_IMPL_H_

template< typename Real,
          typename Device,
          typename Index >
bool tnlSpmvBenchmark< tnlCSRMatrix< Real, Device, Index > >::setup( const tnlCSRMatrix< RealType, tnlHost, IndexType >& matrix )
{
   this->matrix = matrix;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlSpmvBenchmark< tnlCSRMatrix< Real, Device, Index > >::tearDown()
{
   this->matrix.reset();
}

template< typename Real,
          typename Device,
          typename Index >
void tnlSpmvBenchmark< tnlCSRMatrix< Real, Device, Index > >::writeProgress() const
{
}

template< typename Real,
          typename Device,
          typename Index >
void tnlSpmvBenchmark< tnlCSRMatrix< Real, Device, Index > >::writeToLogTable( std::ostream& logFile,
                                                                               const double& csrGflops,
                                                                               const tnlString& inputMtxFile,
                                                                               const tnlCSRMatrix< RealType, tnlHost, IndexType >& csrMatrix,
                                                                               bool writeMatrixInfo  ) const
{

}

#endif /* TNLSPMVBENCHMARK_IMPL_H_ */
