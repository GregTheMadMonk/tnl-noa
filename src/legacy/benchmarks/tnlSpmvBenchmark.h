/***************************************************************************
                          tnlSpmvBenchmark.h  -  description
                             -------------------
    begin                : Dec 29, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLSPMVBENCHMARK_H_
#define TNLSPMVBENCHMARK_H_

#include "tnlSpmvBenchmarkBase.h"
#include <matrices/tnlCSRMatrix.h>


template< typename Matrix >
class tnlSpmvBenchmark
{
};

template< typename Real, typename Device, typename Index >
class tnlSpmvBenchmark< tnlCSRMatrix< Real, Device, Index > > : public tnlSpmvBenchmarkBase< tnlCSRMatrix< Real, Device, Index > >
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   bool setup( const tnlCSRMatrix< RealType, tnlHost, IndexType >& matrix );

   void tearDown();

   void writeProgress() const;

   void writeToLogTable( std::ostream& logFile,
                                    const double& csrGflops,
                                    const tnlString& inputMtxFile,
                                    const tnlCSRMatrix< RealType, tnlHost, IndexType >& csrMatrix,
                                    bool writeMatrixInfo  ) const;
};

#include "tnlSpmvBenchmark_impl.h"

#endif /* TNLSPMVBENCHMARK_H_ */
