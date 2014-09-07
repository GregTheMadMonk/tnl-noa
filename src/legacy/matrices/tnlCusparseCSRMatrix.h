/***************************************************************************
                          tnlCusparseCSRMatrix.h  -  description
                             -------------------
    begin                : Feb 16, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/


#ifndef TNLCUSPARSECSRMATRIX_H
#define TNLCUSPARSECSRMATRIX_H

#include <iostream>
#include <iomanip>
#include <tnlConfig.h>
#include <core/vectors/tnlVector.h>
#include <core/tnlAssert.h>
#include <core/mfuncs.h>
#include <matrices/tnlCSRMatrix.h>
#include <debug/tnlDebug.h>

#ifdef HAVE_CUSPARSE
#include <cusparse.h>
#endif

using namespace std;

//! Wrapper for Cusparse CSR matrix
/*!
 */
template< typename Real, typename Device = tnlHost, typename Index = int  >
class tnlCusparseCSRMatrix : public tnlMatrix< Real, Device, Index >
{
   public:
   //! Basic constructor
   tnlCusparseCSRMatrix( const tnlString& name );

   /*!***
     * Destructor
     */
   ~tnlCusparseCSRMatrix();

   const tnlString& getMatrixClass() const;

   tnlString getType() const;

   Index getCUDABlockSize() const;

   //! This can only be a multiple of the groupSize
   void setCUDABlockSize( Index blockSize );

   //! Sets the number of row and columns.
   bool setSize( Index new_size );

   bool setLike( const tnlCusparseCSRMatrix< Real, Device, Index >& matrix );

   //! Allocate memory for the nonzero elements.
   bool setNonzeroElements( Index elements );

   void reset();

   Index getNonzeroElements() const;

   Index getArtificialZeroElements() const;

   Real getElement( Index row, Index column ) const;

   bool setElement( Index row,
                    Index colum,
                    const Real& value );

   bool addToElement( Index row,
                      Index column,
                      const Real& value );


   bool copyFrom( const tnlCSRMatrix< Real, tnlHost, Index >& csr_matrix );

   template< typename Device2 >
   bool copyFrom( const tnlRgCSRMatrix< Real, Device2, Index >& rgCSRMatrix );

   Real rowProduct( Index row,
                    const tnlVector< Real, Device, Index >& vector ) const;

   void vectorProduct( const tnlVector< Real, Device, Index >& x,
                       tnlVector< Real, Device, Index >& b ) const;

   Real getRowL1Norm( Index row ) const;

   void multiplyRow( Index row, const Real& value );

   //! Prints out the matrix structure
   void printOut( ostream& str,
                  const tnlString& format = tnlString( "" ),
                  const Index lines = 0 ) const;

   bool draw( ostream& str,
              const tnlString& format,
              tnlCSRMatrix< Real, Device, Index >* csrMatrix = 0,
              int verbose = 0 );

   protected:

#ifdef HAVE_CUSPARSE
   cusparseHandle_t   cusparseHandle;
   cusparseMatDescr_t cusparseMatDescr;
#endif

   tnlVector< Real, Device, Index > nonzero_elements;

   tnlVector< Index, Device, Index > columns;

   tnlVector< Index, Device, Index > row_offsets;

};

#ifdef HAVE_CUSPARSE
//TODO: fix this - it does not work with template specialisation
inline void cusparseSpmv( cusparseHandle_t cusparseHandle,
                          cusparseMatDescr_t cusparseMatDescr,
                          int size,
                          const float* nonzeroElements,
                          const int* rowOffsets,
                          const int* columns,
                          const float* x,
                          float* y )
{
   cusparseScsrmv( cusparseHandle,
                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                   size,
                   size,
                   1.0,
                   cusparseMatDescr,
                   nonzeroElements,
                   rowOffsets,
                   columns,
                   x,
                   0.0,
                   y );
}

inline void cusparseSpmv( cusparseHandle_t cusparseHandle,
                          cusparseMatDescr_t cusparseMatDescr,
                          int size,
                          const double* nonzeroElements,
                          const int* rowOffsets,
                          const int* columns,
                          const double* x,
                          double* y )
{
   cusparseDcsrmv( cusparseHandle,
                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                   size,
                   size,
                   1.0,
                   cusparseMatDescr,
                   nonzeroElements,
                   rowOffsets,
                   columns,
                   x,
                   0.0,
                   y );
}

#endif


template< typename Real, typename Device, typename Index >
tnlCusparseCSRMatrix< Real, Device, Index > :: tnlCusparseCSRMatrix( const tnlString& name )
   : tnlMatrix< Real, Device, Index >( name ),
     nonzero_elements( name + " : nonzero-elements" ),
     columns( name + " : columns" ),
     row_offsets( name + " : row_offsets" )
{
#ifdef HAVE_CUSPARSE
   cusparseCreate( &cusparseHandle );
   cusparseCreateMatDescr( &cusparseMatDescr );
   cusparseSetMatType( cusparseMatDescr, CUSPARSE_MATRIX_TYPE_GENERAL );
   cusparseSetMatIndexBase( cusparseMatDescr, CUSPARSE_INDEX_BASE_ZERO );
#endif
};

template< typename Real, typename Device, typename Index >
const tnlString& tnlCusparseCSRMatrix< Real, Device, Index > :: getMatrixClass() const
{
   return tnlMatrixClass :: cusparse;
};

template< typename Real, typename Device, typename Index >
tnlString tnlCusparseCSRMatrix< Real, Device, Index > :: getType() const
{
   return tnlString( "tnlCusparseCSRMatrix< ") +
          tnlString( getType( Real( 0.0 ) ) ) +
          tnlString( ", " ) +
          Device :: getDeviceType() +
          tnlString( " >" );
};

template< typename Real, typename Device, typename Index >
bool tnlCusparseCSRMatrix< Real, Device, Index > :: setSize( Index new_size )
{
   this -> size = new_size;
   if( ! row_offsets. setSize( this -> size + 1 ) )
      return false;
   row_offsets. setValue( 0 );
   return true;
};

template< typename Real, typename Device, typename Index >
bool tnlCusparseCSRMatrix< Real, Device, Index > :: setLike( const tnlCusparseCSRMatrix< Real, Device, Index >& matrix )
{
   dbgFunctionName( "tnlCusparseCSRMatrix< Real, Device, Index >", "setLike" );
   dbgCout( "Setting size to " << matrix. getSize() << "." );

   this -> size = matrix. getSize();
   if( ! nonzero_elements. setLike( matrix. nonzero_elements ) ||
       ! columns. setLike( matrix. columns ) ||
       ! row_offsets. setLike( matrix. row_offsets ) )
      return false;
   row_offsets. setValue( 0 );
   return true;
}

template< typename Real, typename Device, typename Index >
void tnlCusparseCSRMatrix< Real, Device, Index > :: reset()
{
   nonzero_elements. reset();
   columns. reset();
   row_offsets. reset();
}

template< typename Real, typename Device, typename Index >
bool tnlCusparseCSRMatrix< Real, Device, Index > :: setNonzeroElements( Index elements )
{
   if( ! nonzero_elements. setSize( elements ) )
      return false;
   nonzero_elements. setValue( 0 );
   if( ! columns. setSize( elements ) )
      return false;
   columns. setValue( -1 );
   return true;
};

template< typename Real, typename Device, typename Index >
Index tnlCusparseCSRMatrix< Real, Device, Index > :: getNonzeroElements() const
{
   return nonzero_elements. getSize();
}

template< typename Real, typename Device, typename Index >
Index tnlCusparseCSRMatrix< Real, Device, Index > :: getArtificialZeroElements() const
{
   return 0;
}

template< typename Real, typename Device, typename Index >
Real tnlCusparseCSRMatrix< Real, Device, Index > :: getElement( Index row, Index column ) const
{
   tnlAssert( false, );
}

template< typename Real, typename Device, typename Index >
bool tnlCusparseCSRMatrix< Real, Device, Index > :: setElement( Index row, Index column, const Real& v )
{
   tnlAssert( false, );
}

template< typename Real, typename Device, typename Index >
bool tnlCusparseCSRMatrix< Real, Device, Index > :: addToElement( Index row, Index column, const Real& v )
{
   tnlAssert( false, );
}

template< typename Real, typename Device, typename Index >
bool tnlCusparseCSRMatrix< Real, Device, Index > :: copyFrom( const tnlCSRMatrix< Real, tnlHost, Index >& csr_matrix )
{
   if( ! this -> setSize( csr_matrix. getSize() ) ||
       ! this -> setNonzeroElements( csr_matrix. getNonzeroElements() ) )
         return false;

   this -> nonzero_elements = csr_matrix. nonzero_elements;
   this -> columns = csr_matrix. columns;
   this -> row_offsets = csr_matrix. row_offsets;
   return true;
}

template< typename Real, typename Device, typename Index >
Real tnlCusparseCSRMatrix< Real, Device, Index > :: rowProduct( Index row,
                                                                const tnlVector< Real, Device, Index >& vector ) const
{
   abort();
   return 0.0;
}


template< typename Real, typename Device, typename Index >
void tnlCusparseCSRMatrix< Real, Device, Index > :: vectorProduct( const tnlVector< Real, Device, Index >& x,
                                                                   tnlVector< Real, Device, Index >& b ) const
{
#ifdef HAVE_CUSPARSE
  cusparseSpmv( cusparseHandle,
                cusparseMatDescr,
                this -> getSize(),
                this -> nonzero_elements. getData(),
                this -> row_offsets. getData(),
                this -> columns. getData(),
                x. getData(),
                b. getData() );
#endif
}

template< typename Real, typename Device, typename Index >
Real tnlCusparseCSRMatrix< Real, Device, Index > :: getRowL1Norm( Index row ) const
{
   tnlAssert( false, );
   return 0.0;
}

template< typename Real, typename Device, typename Index >
void tnlCusparseCSRMatrix< Real, Device, Index > :: multiplyRow( Index row, const Real& value )
{
   tnlAssert( false, );
}

template< typename Real, typename Device, typename Index >
bool tnlCusparseCSRMatrix< Real, Device, Index > :: draw( ostream& str,
                                                          const tnlString& format,
                                                          tnlCSRMatrix< Real, Device, Index >* csrMatrix,
                                                          int verbose )
{
   tnlAssert( false, );
   return false;
}

template< typename Real, typename Device, typename Index >
void tnlCusparseCSRMatrix< Real, Device, Index > :: printOut( ostream& stream,
                                                              const tnlString& format,
                                                              const Index lines ) const
{

}


template< typename Real, typename Device, typename Index >
tnlCusparseCSRMatrix< Real, Device, Index > :: ~tnlCusparseCSRMatrix()
{
#ifdef HAVE_CUSPARSE
   cusparseDestroyMatDescr( cusparseMatDescr );
   cusparseDestroy( cusparseHandle );
#endif
}

#endif  /* TNLSPMVBENCHMARKCUSPARSEMATRIX_H_ */
