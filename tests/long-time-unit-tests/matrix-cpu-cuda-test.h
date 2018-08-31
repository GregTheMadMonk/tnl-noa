/***************************************************************************
                          matrix-cpu-cuda-test.h  -  description
                             -------------------
    begin                : Aug 31, 2018
    copyright            : (C) 2018 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Config/ConfigDescription.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Matrices/MatrixReader.h>
#include <TNL/Matrices/Matrix.h>
#include <TNL/Matrices/Dense.h>
#include <TNL/Matrices/EllpackSymmetricGraph.h>

using namespace TNL;

void setupConfig( Config::ConfigDescription& config )
{
    config.addDelimiter                            ( "General settings:" );
    config.addRequiredEntry< String >( "input-file" , "Input file name." );
}

template< typename matrix >
bool testCPU( matrix& sparseMatrix, Matrices::Matrix< double, Devices::Host, int >& denseMatrix )
{
    for( int i = 0; i < denseMatrix.getRows(); i++ )
        for( int j = 0; j < denseMatrix.getColumns(); j++ )
        {
            double a = sparseMatrix.getElement( i, j );
            double b = denseMatrix.getElement( i, j );
            if( a != b )
            {
                std::cerr << "Matrices differ on position " << i << " " << j << "." << std::endl;
                std::cerr << "Elements are: "
                     << "\nsparseMatrix.getElement( " << i << ", " << j << " ) == " << sparseMatrix.getElement(i, j)
                     << "\ndenseMatrix.getElement( " << i << ", " << j << " ) == " << denseMatrix.getElement(i, j)
                     << std::endl;
                return 1;
            }
        }
    std::cout << "Elements in sparse and dense matrix are the same. Everything is peachy so far." << std::endl;

    Containers::Vector< double, Devices::Host, int > x, b;
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
                std::cerr << "SPMV gives wrong result! Elements are: "
                     << "\n denseMatrix.getElement(" << j << ", " << i << ") == " << denseMatrix.getElement( j, i )
                     << "\n cudaMatrix.vectorProduct() == " << b.getElement( j ) << std::endl;
                return 1;
            }
    }
    std::cout << "SPMV passed. We can go to GPUs!" << std::endl;

    return 0;
}

template< typename matrix >
bool testGPU( Matrices::Matrix< double, Devices::Host, int >& hostMatrix, matrix& cudaMatrix)
{
    // first perform compare test -- compare all elements using getElement( i, j ) method
    /*for( int i = 0; i < hostMatrix.getRows(); i++ )
        for( int j = 0; j < hostMatrix.getColumns(); j++ )
        {
            double a = hostMatrix.getElement( i, j );
            double b = cudaMatrix.getElement( i, j );
            if( a != b )
            {
               std::cerr << "Matrices differ on position " << i << " " << j << "." << std::endl;
               std::cerr << "Elements are: "
                     << "\nhostMatrix.getElement( " << i << ", " << j << " ) == " << hostMatrix.getElement( i, j )
                     << "\ncudaMatrix.getElement( " << i << ", " << j << " ) == " << cudaMatrix.getElement( i, j )
                     << std::endl;
                return 1;
            }
        }
    std::cout << "Elements in sparse and dense matrix are the same. Everything is peachy so far." << std::endl;*/

    Containers::Vector< double, Devices::Cuda, int > x, b;
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
                std::cerr << "SPMV gives wrong result! Elements are: "
                     << "\n hostMatrix.getElement(" << j << ", " << i << ") == " << hostMatrix.getElement( j, i )
                     << "\n cudaMatrix.vectorProduct() == " << b.getElement( j ) << std::endl;
                return 1;
            }
        if( i % 100 == 0 )
            std::cout << ".";
    }
    std::cout << "SPMV passed. We can go to production!" << std::endl;

    return 0;
}

int main( int argc, char* argv[] )
{
    Config::ParameterContainer parameters;
    Config::ConfigDescription conf_desc;

    setupConfig( conf_desc );

    if( !parseCommandLine( argc, argv, conf_desc, parameters ) )
    {
        conf_desc.printUsage( argv[ 0 ] );
        return 1;
    }

    const String& inputFile = parameters.getParameter< String >( "input-file" );

    typedef Matrices::EllpackSymmetricGraph< double, Devices::Host, int > EllpackSymmetricGraphHost;
    EllpackSymmetricGraphHost hostMatrix;
    if( !Matrices::MatrixReader< EllpackSymmetricGraphHost >::readMtxFile( inputFile, hostMatrix, true, true ) )
        return 1;
    if( !hostMatrix.help( true ) )
        return 1;

    typedef Matrices::Dense< double, Devices::Host, int > DenseMatrix;
    DenseMatrix denseMatrix;
    if( ! Matrices::MatrixReader< DenseMatrix >::readMtxFile( inputFile, denseMatrix, true ) )
        return false;

    if( testCPU< Matrices::EllpackSymmetricGraph< double, Devices::Host, int > >( hostMatrix, denseMatrix ) == 1 )
        return 1;

    typedef Matrices::EllpackSymmetricGraph< double, Devices::Cuda, int > EllpackSymmetricGraphCuda;
    EllpackSymmetricGraphCuda cudaMatrix;
    cudaMatrix.copyFromHostToCuda( hostMatrix );

    return testGPU< Matrices::EllpackSymmetricGraph< double, Devices::Cuda, int > >( denseMatrix, cudaMatrix );
}

