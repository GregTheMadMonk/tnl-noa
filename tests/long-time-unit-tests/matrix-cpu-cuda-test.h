#ifndef MATRIX_CPU_CUDA_TEST_H_
#define MATRIX_CPU_CUDA_TEST_H_

#include <config/tnlConfigDescription.h>
#include <config/tnlParameterContainer.h>
#include <matrices/tnlMatrixReader.h>
#include <matrices/tnlMatrix.h>
#include <matrices/tnlDenseMatrix.h>
#include <matrices/tnlEllpackGraphMatrix.h>

void setupConfig( tnlConfigDescription& config )
{
    config.addDelimiter                            ( "General settings:" );
    config.addRequiredEntry< tnlString >( "input-file" , "Input file name." );
}

template< typename matrix >
bool testCPU( matrix& sparseMatrix, tnlMatrix< double, tnlHost, int >& denseMatrix )
{
    for( int i = 0; i < denseMatrix.getRows(); i++ )
        for( int j = 0; j < denseMatrix.getColumns(); j++ )
        {
            double a = sparseMatrix.getElement( i, j );
            double b = denseMatrix.getElement( i, j );
            if( a != b )
            {
                cerr << "Matrices differ on position " << i << " " << j << "." << endl;
                cerr << "Elements are: "
                     << "\nsparseMatrix.getElement( " << i << ", " << j << " ) == " << sparseMatrix.getElement(i, j)
                     << "\ndenseMatrix.getElement( " << i << ", " << j << " ) == " << denseMatrix.getElement(i, j)
                     << endl;
                return 1;
            }
        }
    cout << "Elements in sparse and dense matrix are the same. Everything is peachy so far." << endl;

    tnlVector< double, tnlHost, int > x, b;
    x.setSize( denseMatrix.getColumns() );
    b.setSize( denseMatrix.getRows() );
    for( int i = 0; i < x.getSize(); i++ )
    {
        b.setValue( 0.0 );
        x.setValue( 0.0 );
        x.setElement( i, 1.0 );
        sparseMatrix.vectorProduct( x, b );
        for( int j = 0; j < b.getSize(); j++ )
            if( b.getElement( j ) != sparseMatrix.getElement( j, i ) )
            {
                cerr << "SPMV gives wrong result! Elements are: "
                     << "\n denseMatrix.getElement(" << j << ", " << i << ") == " << denseMatrix.getElement( j, i )
                     << "\n cudaMatrix.vectorProduct() == " << b.getElement( j ) << endl;
                return 1;
            }
    }
    cout << "SPMV passed. We can go to GPUs!" << endl;

    return 0;
}

template< typename matrix >
bool testGPU( tnlMatrix< double, tnlHost, int >& hostMatrix, matrix& cudaMatrix)
{
    // first perform compare test -- compare all elements using getElement( i, j ) method
    for( int i = 0; i < hostMatrix.getRows(); i++ )
        for( int j = 0; j < hostMatrix.getColumns(); j++ )
        {
            double a = hostMatrix.getElement( i, j );
            double b = cudaMatrix.getElement( i, j );
            if( a != b )
            {
                cerr << "Matrices differ on position " << i << " " << j << "." << endl;
                cerr << "Elements are: "
                     << "\nhostMatrix.getElement( " << i << ", " << j << " ) == " << hostMatrix.getElement( i, j )
                     << "\ncudaMatrix.getElement( " << i << ", " << j << " ) == " << cudaMatrix.getElement( i, j )
                     << endl;
                return 1;
            }
        }
    cout << "Elements in sparse and dense matrix are the same. Everything is peachy so far." << endl;

    tnlVector< double, tnlCuda, int > x, b;
    x.setSize( cudaMatrix.getColumns() );
    b.setSize( cudaMatrix.getRows() );
    for( int i = 0; i < x.getSize(); i++ )
    {
        b.setValue( 0.0 );
        x.setValue( 0.0 );
        x.setElement( i, 1.0 );
        cudaMatrix.vectorProduct( x, b );
        for( int j = 0; j < b.getSize(); j++ )
            if( b.getElement( j ) != hostMatrix.getElement( j, i ) )
            {
                cerr << "SPMV gives wrong result! Elements are: "
                     << "\n hostMatrix.getElement(" << j << ", " << i << ") == " << hostMatrix.getElement( j, i )
                     << "\n cudaMatrix.vectorProduct() == " << b.getElement( j ) << endl;
                return 1;
            }
    }
    cout << "SPMV passed. We can go to production!" << endl;

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
    if( !hostMatrix.help( true ) )
        return 1;

    typedef tnlDenseMatrix< double, tnlHost, int > DenseMatrix;
    DenseMatrix denseMatrix;
    if( ! tnlMatrixReader< DenseMatrix >::readMtxFile( inputFile, denseMatrix, true ) )
        return false;

    if( testCPU< tnlEllpackGraphMatrix< double, tnlHost, int > >( hostMatrix, denseMatrix ) == 1 )
        return 1;

    typedef tnlEllpackGraphMatrix< double, tnlCuda, int > EllpackGraphCuda;
    EllpackGraphCuda cudaMatrix;
    cudaMatrix.copyFromHostToCuda( hostMatrix );

    return testGPU< tnlEllpackGraphMatrix< double, tnlCuda, int > >( denseMatrix, cudaMatrix );
}

#endif // MATRIX_CPU_CUDA_TEST_H_