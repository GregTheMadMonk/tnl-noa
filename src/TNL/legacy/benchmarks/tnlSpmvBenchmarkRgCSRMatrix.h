 /***************************************************************************
                          tnlSpmvBenchmarkRgCSRMatrix.h  -  description
                             -------------------
    begin                : May 15, 2011
    copyright            : (C) 2011 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLSPMVBENCHMARKRGCSRMATRIX_H_
#define TNLSPMVBENCHMARKRGCSRMATRIX_H_

#include "tnlSpmvBenchmark.h"

template< typename Real, typename Device, typename Index>
class tnlSpmvBenchmarkRgCSRMatrix : public tnlSpmvBenchmark< Real, Device, Index, tnlRgCSRMatrix >
{
   public:

   tnlSpmvBenchmarkRgCSRMatrix();

   bool setup( const tnlCSRMatrix< Real, Devices::Host, Index >& matrix );

   void tearDown();

   void writeProgress() const;

   void writeToLogTable( std::ostream& logFile,
                         const double& csrGflops,
                         const String& inputMtxFile,
                         const tnlCSRMatrix< Real, Devices::Host, Index >& csrMatrix,
                         bool writeMatrixInfo ) const;

   void setGroupSize( const Index groupSize );

   void setCudaBlockSize( const Index cudaBlockSize );

   void setUseAdaptiveGroupSize( bool useAdaptiveGroupSize );

   void setAdaptiveGroupSizeStrategy( tnlAdaptiveGroupSizeStrategy adaptiveGroupSizeStrategy );

   Index getArtificialZeroElements() const;

   protected:

   Index groupSize;

   Index cudaBlockSize;

   bool useAdaptiveGroupSize;

   tnlAdaptiveGroupSizeStrategy adaptiveGroupSizeStrategy;
};

template< typename Real,
          typename Device,
          typename Index>
tnlSpmvBenchmarkRgCSRMatrix< Real, Device, Index > :: tnlSpmvBenchmarkRgCSRMatrix()
 : groupSize( 0 ),
   cudaBlockSize( 0 ),
   useAdaptiveGroupSize( false ),
   adaptiveGroupSizeStrategy( tnlAdaptiveGroupSizeStrategyByAverageRowSize )
{
}

template< typename Real,
          typename Device,
          typename Index>
bool tnlSpmvBenchmarkRgCSRMatrix< Real, Device, Index > :: setup( const tnlCSRMatrix< Real, Devices::Host, Index >& csrMatrix )
{
   Assert( this->groupSize > 0, std::cerr << "groupSize = " << this->groupSize );
   if( Device :: getDevice() == Devices::HostDevice )
   {
      this->matrix. tuneFormat( groupSize,
                                  this->useAdaptiveGroupSize,
                                  this->adaptiveGroupSizeStrategy );
      if( ! this->matrix. copyFrom( csrMatrix ) )
         return false;
   }
   if( Device :: getDevice() == Devices::CudaDevice )
   {
#ifdef HAVE_CUDA
      tnlRgCSRMatrix< Real, Devices::Host, Index > hostMatrix( "tnlSpmvBenchmarkRgCSRMatrix< Real, Device, Index > :: setup : hostMatrix" );
      hostMatrix. tuneFormat( groupSize,
                              this->useAdaptiveGroupSize,
                              this->adaptiveGroupSizeStrategy );
      hostMatrix. copyFrom( csrMatrix );
      if( ! this->matrix. copyFrom( hostMatrix ) )
         return false;
#else
      return false;
#endif
   }
   this->setupOk = true;
   return true;
}

template< typename Real,
          typename Device,
          typename Index>
void tnlSpmvBenchmarkRgCSRMatrix< Real, Device, Index > :: tearDown()
{
   this->matrix. reset();
}

template< typename Real,
          typename Device,
          typename Index >
void tnlSpmvBenchmarkRgCSRMatrix< Real, Device, Index > :: writeProgress() const
{
  std::cout << left << std::setw( this->formatColumnWidth - 15 ) << "Row-grouped CSR ";
   if( Device :: getDevice() == Devices::CudaDevice )
   {
      if( useAdaptiveGroupSize )
        std::cout << std::setw( 5 ) << "Var.";
      else
        std::cout << std::setw( 5 ) << this->groupSize;
     std::cout << std::setw( 10 ) << this->cudaBlockSize;
   }
   else
   {
      if( useAdaptiveGroupSize )
        std::cout << std::setw( 15 ) << "Var.";
      else
        std::cout << std::setw( 15 ) << this->groupSize;
   }
  std::cout << right << std::setw( this->timeColumnWidth ) << std::setprecision( 2 ) << this->getTime()
        << right << std::setw( this->iterationsColumnWidth ) << this->getIterations()
        << right << std::setw( this->gflopsColumnWidth ) << std::setprecision( 2 ) << this->getGflops();
   if( this->getBenchmarkWasSuccesful() )
       std::cout << right << std::setw( this->benchmarkStatusColumnWidth ) << "  OK  - maxError is " << this->maxError << ". ";
   else
       std::cout << right << std::setw( this->benchmarkStatusColumnWidth ) << "  FAILED - maxError is " << this->maxError << ". ";
#ifndef HAVE_CUDA
   if( Device :: getDevice() == Devices::CudaDevice )
      CudaSupportMissingMessage;;
#endif
     std::cout << std::endl;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlSpmvBenchmarkRgCSRMatrix< Real, Device, Index > :: setGroupSize( const Index groupSize )
{
   this->groupSize = groupSize;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlSpmvBenchmarkRgCSRMatrix< Real, Device, Index > :: setCudaBlockSize( const Index cudaBlockSize )
{
   this->matrix. setCUDABlockSize( cudaBlockSize );
   this->cudaBlockSize = cudaBlockSize;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlSpmvBenchmarkRgCSRMatrix< Real, Device, Index > :: setUseAdaptiveGroupSize( bool useAdaptiveGroupSize )
{
   this->useAdaptiveGroupSize = useAdaptiveGroupSize;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlSpmvBenchmarkRgCSRMatrix< Real, Device, Index > :: setAdaptiveGroupSizeStrategy( tnlAdaptiveGroupSizeStrategy adaptiveGroupSizeStrategy )
{
   this->adaptiveGroupSizeStrategy = adaptiveGroupSizeStrategy;
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlSpmvBenchmarkRgCSRMatrix< Real, Device, Index > :: getArtificialZeroElements() const
{
   return this->matrix. getArtificialZeroElements();
}

template< typename Real,
          typename Device,
          typename Index >
void tnlSpmvBenchmarkRgCSRMatrix< Real, Device, Index > :: writeToLogTable( std::ostream& logFile,
                                                                            const double& csrGflops,
                                                                            const String& inputMtxFile,
                                                                            const tnlCSRMatrix< Real, Devices::Host, Index >& csrMatrix,
                                                                            bool writeMatrixInfo ) const
{
   String bgColor;
   switch( groupSize )
   {
      case 16: bgColor = "#5555FF"; break;
      case 32: bgColor = "#9999FF"; break;
      case 64: bgColor = "#CCCCFF"; break;
      default: bgColor = "#FFFFFF";
   }
   if( writeMatrixInfo )
   {
      String baseFileName( inputMtxFile );
      baseFileName += String( ".rgcsr-");
      baseFileName += String( groupSize );
      String matrixPdfFile( baseFileName );
      matrixPdfFile += String( ".pdf" );
      String matrixHtmlFile( baseFileName );
      matrixHtmlFile += String( ".html" );
      tnlRgCSRMatrix< Real > rgCsrMatrix( inputMtxFile );
      rgCsrMatrix. tuneFormat( this->groupSize,
                               this->useAdaptiveGroupSize,
                               this->adaptiveGroupSizeStrategy );
      rgCsrMatrix. copyFrom( csrMatrix );
      this->printMatrixInHtml( matrixHtmlFile, rgCsrMatrix );
      logFile << "             <td bgcolor=" << bgColor << "> <a href=\"" << matrixPdfFile << "\">PDF</a>,<a href=\"" << matrixHtmlFile << "\"> HTML</a></td>" << std::endl;
      logFile << "             <td bgcolor=" << bgColor << "> " << this->getArtificialZeroElements() << "</td>" << std::endl;
   }
   if( this->getBenchmarkWasSuccesful() )
   {
      const double speedUp = this->getGflops() / csrGflops;
      bgColor =  this->getBgColorBySpeedUp( speedUp );
      logFile << "             <td bgcolor=" << bgColor << ">" << this->getTime() << "</td>" << std::endl;
      logFile << "             <td bgcolor=" << bgColor << "> " << this->getGflops() << "</td>" << std::endl;
      logFile << "             <td bgcolor=" << bgColor << "> " << speedUp << "</td>" << std::endl;
   }
   else
   {
      logFile << "             <td bgcolor=#FF0000> N/A </td>" << std::endl;
      logFile << "             <td bgcolor=#FF0000> N/A </td>" << std::endl;
      logFile << "             <td bgcolor=#FF0000> N/A </td>" << std::endl;
   }
}


#endif /* TNLSPMVBENCHMARKRGCSRMATRIX_H_ */
