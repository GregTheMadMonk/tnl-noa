#ifndef MATRIX_CPU_CUDA_TEST_H_
#define MATRIX_CPU_CUDA_TEST_H_

#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>
#include <matrices/tnlMatrixReader.h>
#include <matrices/tnlMatrix.h>
#include <matrices/tnlEllpackGraphMatrix.h>

void setupConfig( tnlConfigDescription& config )
{
    config.addDelimiter                            ( "General settings:" );
    config.addRequiredEntry< tnlString >( "input-file" , "Input file name." );
}

int test( tnlMatrix< double, tnlHost, int >& hostMatrix, tnlMatrix< double, tnlCuda, int >& cudaMatrix )
{
    // first perform compare test -- compare all elements using getElement( i, j ) method
    for( int i = 0; i < hostMatrix.getRows(); i++ )
        for( int j = 0; j < hostMatrix.getColumns(); j++ )
        {
            if( hostMatrix.getElement( i, j ) != cudaMatrix.getElement( i, j ) )
            {
                cerr << "Matrices differ on position " << i << " " << j << "." << endl;
                cerr << "Elements are: "
                     << "\nhostMatrix.getElement( " << i << ", " << j << " ) == " << hostMatrix.getElement( i, j )
                     << "\ncudaMatrix.getElement( " << i << ", " << j << " ) == " << cudaMatrix.getElement( i, j )
                     << endl;
                return 1;
            }
        }
    return 0;
}

int main( int argc, char* argv[] )
{
    tnlParameterContainer parameters;
    tnlConfigDescription conf_desc;

    setupConfig( conf_desc );

    if( !ParseCommandLine( argc, argv, conf_desc, parameters ) )
    {
        conf_desc.printUsage( argv[ 0 ] );
        return 1;
    }

    const tnlString& inputFile = parameters.GetParameter< tnlString >( "input-file" );

    typedef tnlEllpackGraphMatrix< double, tnlHost, int > EllpackGraphHost;
    EllpackGraphHost hostMatrix;
    if( !tnlMatrixReader< EllpackGraphHost >::readMtxFile( inputFile, hostMatrix, true, true ) )
        return 1;

    typedef tnlEllpackGraphMatrix< double, tnlCuda, int > EllpackGraphCuda;
    EllpackGraphCuda cudaMatrix;
    if( !cudaMatrix.copyFromHostToCuda( hostMatrix ) )
        return 1;

    return test( hostMatrix, cudaMatrix );
}

#endif // MATRIX_CPU_CUDA_TEST_H_