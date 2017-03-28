/***************************************************************************
                          tnlSpmvBenchmarkAdaptiveRgCSR.h  -  description
                             -------------------
    begin                : May 15, 2011
    copyright            : (C) 2011 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef TNLSPMVBENCHMARKADAPTIVERGCSRMATRIX_H_
#define TNLSPMVBENCHMARKADAPTIVERGCSRMATRIX_H_

#include "tnlSpmvBenchmark.h"
#include <TNL/Assert.h>

template< typename Real, typename Device, typename Index>
class tnlSpmvBenchmarkAdaptiveRgCSR : public tnlSpmvBenchmark< Real, Device, Index, tnlAdaptiveRgCSR >
{
   public:

   tnlSpmvBenchmarkAdaptiveRgCSR();

   bool setup( const CSR< Real, Devices::Host, Index >& matrix );

   void tearDown();

   void writeProgress() const;

   void writeToLogTable( std::ostream& logFile,
                         const double& csrGflops,
                         const String& inputMtxFile,
                         const CSR< Real, Devices::Host, Index >& csrMatrix,
                         bool writeMatrixInfo  ) const;

   void setDesiredChunkSize( const Index desiredChunkSize );

   void setCudaBlockSize( const Index cudaBlockSize );

   Index getArtificialZeroElements() const;

   void setBestRgCSRGflops( const double& bestRgCSRGflops );

   protected:

   /****
    * This is helper method for generating HTML table with benchmark results
    */
    String getBgColorByRgCSRSpeedUp( const double& speedUp ) const;

   Index desiredChunkSize;

   Index cudaBlockSize;

   bool useAdaptiveGroupSize;

   tnlAdaptiveGroupSizeStrategy adaptiveGroupSizeStrategy;

   double bestRgCSRGflops;
};

template< typename Real,
          typename Device,
          typename Index>
tnlSpmvBenchmarkAdaptiveRgCSR< Real, Device, Index > :: tnlSpmvBenchmarkAdaptiveRgCSR()
 : desiredChunkSize( 4 ),
   cudaBlockSize( 32 ),
   useAdaptiveGroupSize( false ),
   adaptiveGroupSizeStrategy( tnlAdaptiveGroupSizeStrategyByAverageRowSize ),
   bestRgCSRGflops( 0.0 )

{
}

template< typename Real,
          typename Device,
          typename Index>
bool tnlSpmvBenchmarkAdaptiveRgCSR< Real, Device, Index > :: setup( const CSR< Real, Devices::Host, Index >& matrix )
{
   //TNL_ASSERT( this->groupSize > 0, std::cerr << "groupSize = " << this->groupSize );
   if( Device :: getDevice() == Devices::HostDevice )
   {
      this->matrix. tuneFormat( desiredChunkSize, cudaBlockSize );
      if( ! this->matrix. copyFrom( matrix ) )
         return false;
      //matrix. printOut(std::cout, "text", 30 );
      //this->matrix. printOut(std::cout, "text", 30 );
   }
   if( Device :: getDevice() == Devices::CudaDevice )
   {
#ifdef HAVE_CUDA
      tnlAdaptiveRgCSR< Real, Devices::Host, Index > hostMatrix( "tnlSpmvBenchmarkAdaptiveRgCSR< Real, Device, Index > :: setup : hostMatrix" );
      hostMatrix. tuneFormat( desiredChunkSize, cudaBlockSize );
      hostMatrix. copyFrom( matrix );
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
void tnlSpmvBenchmarkAdaptiveRgCSR< Real, Device, Index > :: tearDown()
{
   //this->matrix. setSize( 0 );
   //this->matrix. setNonzeroElements( 0 );
}

template< typename Real,
          typename Device,
          typename Index >
void tnlSpmvBenchmarkAdaptiveRgCSR< Real, Device, Index > :: writeProgress() const
{
  std::cout << left << std::setw( this->formatColumnWidth - 15 ) << "Adap. Row-grouped CSR ";
   if( Device :: getDevice() == Devices::CudaDevice )
     std::cout << std::setw( 5 ) << this->desiredChunkSize
           << std::setw( 10 ) << this->cudaBlockSize;
   else
     std::cout << std::setw( 15 ) << this->desiredChunkSize;
  std::cout << right << std::setw( this->timeColumnWidth ) << std::setprecision( 2 ) << this->getTime()
        << right << std::setw( this->iterationsColumnWidth ) << this->getIterations()
        << right << std::setw( this->gflopsColumnWidth ) << std::setprecision( 2 ) << this->getGflops();
   if( this->getBenchmarkWasSuccesful() )
       std::cout << right << std::setw( this->benchmarkStatusColumnWidth ) << " OK - maxError is " << this->maxError << ". ";
   else
       std::cout << right << std::setw( this->benchmarkStatusColumnWidth ) << "  FAILED";
#ifndef HAVE_CUDA
   if( Device :: getDevice() == Devices::CudaDevice )
      CudaSupportMissingMessage;;
#endif
     std::cout << std::endl;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlSpmvBenchmarkAdaptiveRgCSR< Real, Device, Index > :: writeToLogTable( std::ostream& logFile,
                                                                                    const double& csrGflops,
                                                                                    const String& inputMtxFile,
                                                                                    const CSR< Real, Devices::Host, Index >& csrMatrix,
                                                                                    bool writeMatrixInfo  ) const
{
   if( this->getBenchmarkWasSuccesful() )
   {
      String bgColor="#FFFFFF";
      double speedUp = this->getGflops() / csrGflops;
      double rgCsrSpeedUp( 0.0 );
      if( this->bestRgCSRGflops )
         rgCsrSpeedUp = this->getGflops() / this->bestRgCSRGflops;
      switch( desiredChunkSize )
      {
         case 1: bgColor = "#666666"; break;
         case 2: bgColor = "#777777"; break;
         case 4: bgColor = "#888888"; break;
         case 8: bgColor = "#999999"; break;
         case 16: bgColor = "#AAAAAA"; break;
         case 32: bgColor = "#BBBBBB"; break;
         default: bgColor = "#FFFFFF";
      }
      if( writeMatrixInfo )
      {
         String baseFileName( inputMtxFile );
         baseFileName += String( ".argcsr-");
         baseFileName += String( desiredChunkSize );
         baseFileName += String( "-" );
         baseFileName += String( cudaBlockSize );
         String matrixPdfFile = baseFileName + String( ".pdf" );
         String matrixHtmlFile = baseFileName + String( ".html" );
         tnlAdaptiveRgCSR< Real > argCsrMatrix( inputMtxFile );
         argCsrMatrix. tuneFormat( this->desiredChunkSize,
                                 this->cudaBlockSize );
         argCsrMatrix. copyFrom( csrMatrix );
         this->printMatrixInHtml( matrixHtmlFile, argCsrMatrix );
         if( rgCsrSpeedUp > 1.0 )
            bgColor=getBgColorByRgCSRSpeedUp( rgCsrSpeedUp );
         logFile << "             <td bgcolor=" << bgColor << "> <a href=\"" << matrixPdfFile << "\">PDF</a>, <a href=\"" << matrixHtmlFile << "\">HTML</a></td> " << std::endl;
         logFile << "             <td bgcolor=" << bgColor << "> " << this->getArtificialZeroElements() << "</td>" << std::endl;
      }

      bgColor = this->getBgColorBySpeedUp( speedUp );
      String textColor = "#000000"; //getBgColorByRgCSRSpeedUp( rgCsrSpeedUp );
      logFile << "             <td bgcolor=" << bgColor << "><font size=3 color=\"" << textColor << "\"> " << this->getTime() << "</font></td>" << std::endl;
      logFile << "             <td bgcolor=" << bgColor << "><font size=3 color=\"" << textColor << "\"> " << this->getGflops() << "</font></td>" << std::endl;
      logFile << "             <td bgcolor=" << bgColor << "><font size=3 color=\"" << textColor << "\"> " << speedUp << "</font></td>" << std::endl;

   }
   else
   {
      if( writeMatrixInfo )
      {
         logFile << "             <td bgcolor=#FF0000> N/A </td>" << std::endl;
         logFile << "             <td bgcolor=#FF0000> N/A </td>" << std::endl;
      }
      logFile << "             <td bgcolor=#FF0000> N/A </td>" << std::endl;
      logFile << "             <td bgcolor=#FF0000> N/A </td>" << std::endl;
      logFile << "             <td bgcolor=#FF0000> N/A </td>" << std::endl;
   }
}

template< typename Real,
          typename Device,
          typename Index >
void tnlSpmvBenchmarkAdaptiveRgCSR< Real, Device, Index > :: setDesiredChunkSize( const Index desiredChunkSize )
{
   this->desiredChunkSize = desiredChunkSize;
}

template< typename Real,
          typename Device,
          typename Index >
void tnlSpmvBenchmarkAdaptiveRgCSR< Real, Device, Index > :: setCudaBlockSize( const Index cudaBlockSize )
{
   this->cudaBlockSize = cudaBlockSize;
}

template< typename Real,
          typename Device,
          typename Index >
Index tnlSpmvBenchmarkAdaptiveRgCSR< Real, Device, Index > :: getArtificialZeroElements() const
{
   return this->matrix. getArtificialZeroElements();
}

template< typename Real,
          typename Device,
          typename Index >
void tnlSpmvBenchmarkAdaptiveRgCSR< Real, Device, Index > :: setBestRgCSRGflops( const double& bestRgCSRGflops )
{
   this->bestRgCSRGflops = bestRgCSRGflops;
}

template< typename Real,
          typename Device,
          typename Index >
String tnlSpmvBenchmarkAdaptiveRgCSR< Real, Device, Index > :: getBgColorByRgCSRSpeedUp( const double& speedUp ) const
{
   if( speedUp >= 30.0 )
      return String( "#009900" );
   if( speedUp >= 25.0 )
      return String( "#00AA00" );
   if( speedUp >= 20.0 )
      return String( "#00BB00" );
   if( speedUp >= 15.0 )
      return String( "#00CC00" );
   if( speedUp >= 10.0 )
      return String( "#00DD00" );
   if( speedUp >= 5.0 )
      return String( "#00EE00" );
   if( speedUp >= 1.0 )
      return String( "#00FF00" );
   return String( "#FFFFFF" );
}

#endif /* TNLSPMVBENCHMARKADAPTIVERGCSRMATRIX_H_ */
