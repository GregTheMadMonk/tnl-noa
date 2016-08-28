/***************************************************************************
                          tnlSpmvBenchmarkBase.h  -  description
                             -------------------
    begin                : May 15, 2011
    copyright            : (C) 2011 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLSPMVBENCHMARKBASE_H_
#define TNLSPMVBENCHMARKBASE_H_

#include <TNL/Matrices/CSR.h>
#include <TNL/TimerRT.h>
#include <TNL/mfuncs.h>


double tnlSpmvBenchmarkPrecision( const double& ) { return 1.0e-12; }
float tnlSpmvBenchmarkPrecision( const float& ) { return 1.0e-4; }

template< typename Matrix >
class tnlSpmvBenchmarkBase
{
   public:

   tnlSpmvBenchmarkBase();
 
   typedef typename Matrix::RealType RealType;
   typedef typename Matrix::DeviceType DeviceType;
   typedef typename Matrix::IndexType IndexType;

   bool getBenchmarkWasSuccesful() const;

   double getGflops() const;

   double getTime() const;

   void setMaxIterations( const int maxIterations );

   int getIterations() const;

   IndexType getArtificialZeros() const;

   RealType getMaxError() const;

   void writeProgressTableHeader();

   virtual bool setup( const CSR< RealType, Devices::Host, IndexType >& matrix ) = 0;

   virtual void tearDown() = 0;

   virtual void writeProgress() const = 0;

   /****
    * This is virtual only the purpose of testing external formats like
    * the Hybrid format from the CUSP library. This format is not wrapped
    * in Matrix.
    */
   virtual void runBenchmark( const Vector< RealType, DeviceType, IndexType >& x,
                              const Vector< RealType, Devices::Host, IndexType >& refB,
                              bool verbose );

   virtual void writeToLogTable( std::ostream& logFile,
                                 const double& csrGflops,
                                 const String& inputMtxFile,
                                 const CSR< RealType, Devices::Host, IndexType >& csrMatrix,
                                 bool writeMatrixInfo  ) const = 0;

   protected:

   /****
    * This is helper method for generating HTML table with benchmark results
    */
   String getBgColorBySpeedUp( const double& speedUp ) const;

   /****
    * Helper method for writing matrix statistics and information to HTML
    */
   bool printMatrixInHtml( const String& fileName,
                           Matrix< RealType, Devices::Host, IndexType >& matrix ) const;


   bool benchmarkWasSuccesful;

   bool setupOk;

   double gflops;

   double time;

   /****
    * Max number of SpMV repetitions.
    */
   int maxIterations;

   /****
    * Real number of repetitions.
    */
   int iterations;

   IndexType artificialZeros;

   RealType maxError;

   IndexType firstErrorOccurence;

   Matrix matrix;

   /****
    * Parameters for the progress table columns
    */

   int formatColumnWidth;

   int timeColumnWidth;

   int iterationsColumnWidth;

   int gflopsColumnWidth;

   int benchmarkStatusColumnWidth;

   int infoColumnWidth;
};


#include "tnlSpmvBenchmarkBase_impl.h"
#endif /* TNLSPMVBENCHMARKBASE_H_ */
