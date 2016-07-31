/***************************************************************************
                          tnlCusparseCSRMatrix.h  -  description
                             -------------------
    begin                : Feb 16, 2012
    copyright            : (C) 2012 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */


#ifndef TNLCUSPARSECSRMATRIX_H
#define TNLCUSPARSECSRMATRIX_H

#include <iostream>
#include <iomanip>
#include <TNL/tnlConfig.h>
#include <TNL/Vectors/Vector.h>
#include <TNL/Assert.h>
#include <TNL/core/mfuncs.h>
#include <TNL/matrices/tnlCSRMatrix.h>
#include <TNL/debug/tnlDebug.h>

#ifdef HAVE_CUSPARSE
#include <cusparse.h>
#endif

//! Wrapper for Cusparse CSR matrix
/*!
 */
template< typename Real, typename Device = Devices::Host, typename Index = int  >
class tnlCusparseCSRMatrix : public tnlMatrix< Real, Device, Index >
{
   public:
   //! Basic constructor
   tnlCusparseCSRMatrix( const String& name );

   /*!***
     * Destructor
     */
   ~tnlCusparseCSRMatrix();

   const String& getMatrixClass() const;

   String getType() const;

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


   bool copyFrom( const tnlCSRMatrix< Real, Devices::Host, Index >& csr_matrix );

   template< typename Device2 >
   bool copyFrom( const tnlRgCSRMatrix< Real, Device2, Index >& rgCSRMatrix );

   Real rowProduct( Index row,
                    const Vector< Real, Device, Index >& vector ) const;

   void vectorProduct( const Vector< Real, Device, Index >& x,
                       Vector< Real, Device, Index >& b ) const;

   Real getRowL1Norm( Index row ) const;

   void multiplyRow( Index row, const Real& value );

   //! Prints out the matrix structure
   void printOut( std::ostream& str,
                  const String& format = String( "" ),
                  const Index lines = 0 ) const;

   bool draw( std::ostream& str,
              const String& format,
              tnlCSRMatrix< Real, Device, Index >* csrMatrix = 0,
              int verbose = 0 );

   protected:

#ifdef HAVE_CUSPARSE
   cusparseHandle_t   cusparseHandle;
   cusparseMatDescr_t cusparseMatDescr;
#endif

   Vector< Real, Device, Index > nonzero_elements;

   Vector< Index, Device, Index > columns;

   Vector< Index, Device, Index > row_offsets;

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
tnlCusparseCSRMatrix< Real, Device, Index > :: tnlCusparseCSRMatrix( const String& name )
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
const String& tnlCusparseCSRMatrix< Real, Device, Index > :: getMatrixClass() const
{
   return tnlMatrixClass :: cusparse;
};

template< typename Real, typename Device, typename Index >
String tnlCusparseCSRMatrix< Real, Device, Index > :: getType() const
{
   return String( "tnlCusparseCSRMatrix< ") +
          String( getType( Real( 0.0 ) ) ) +
          String( ", " ) +
          Device :: getDeviceType() +
          String( " >" );
};

template< typename Real, typename Device, typename Index >
bool tnlCusparseCSRMatrix< Real, Device, Index > :: setSize( Index new_size )
{
   this->size = new_size;
   if( ! row_offsets. setSize( this->size + 1 ) )
      return false;
   row_offsets. setValue( 0 );
   return true;
};

template< typename Real, typename Device, typename Index >
bool tnlCusparseCSRMatrix< Real, Device, Index > :: setLike( const tnlCusparseCSRMatrix< Real, Device, Index >& matrix )
{
   dbgFunctionName( "tnlCusparseCSRMatrix< Real, Device, Index >", "setLike" );
   dbgCout( "Setting size to " << matrix. getSize() << "." );

   this->size = matrix. getSize();
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
   Assert( false, );
}

template< typename Real, typename Device, typename Index >
bool tnlCusparseCSRMatrix< Real, Device, Index > :: setElement( Index row, Index column, const Real& v )
{
   Assert( false, );
}

template< typename Real, typename Device, typename Index >
bool tnlCusparseCSRMatrix< Real, Device, Index > :: addToElement( Index row, Index column, const Real& v )
{
   Assert( false, );
}

template< typename Real, typename Device, typename Index >
bool tnlCusparseCSRMatrix< Real, Device, Index > :: copyFrom( const tnlCSRMatrix< Real, Devices::Host, Index >& csr_matrix )
{
   if( ! this->setSize( csr_matrix. getSize() ) ||
       ! this->setNonzeroElements( csr_matrix. getNonzeroElements() ) )
         return false;

   this->nonzero_elements = csr_matrix. nonzero_elements;
   this->columns = csr_matrix. columns;
   this->row_offsets = csr_matrix. row_offsets;
   return true;
}

template< typename Real, typename Device, typename Index >
Real tnlCusparseCSRMatrix< Real, Device, Index > :: rowProduct( Index row,
                                                                const Vector< Real, Device, Index >& vector ) const
{
   abort();
   return 0.0;
}


template< typename Real, typename Device, typename Index >
void tnlCusparseCSRMatrix< Real, Device, Index > :: vectorProduct( const Vector< Real, Device, Index >& x,
                                                                   Vector< Real, Device, Index >& b ) const
{
#ifdef HAVE_CUSPARSE
  cusparseSpmv( cusparseHandle,
                cusparseMatDescr,
                this->getSize(),
                this->nonzero_elements. getData(),
                this->row_offsets. getData(),
                this->columns. getData(),
                x. getData(),
                b. getData() );
#endif
}

template< typename Real, typename Device, typename Index >
Real tnlCusparseCSRMatrix< Real, Device, Index > :: getRowL1Norm( Index row ) const
{
   Assert( false, );
   return 0.0;
}

template< typename Real, typename Device, typename Index >
void tnlCusparseCSRMatrix< Real, Device, Index > :: multiplyRow( Index row, const Real& value )
{
   Assert( false, );
}

template< typename Real, typename Device, typename Index >
bool tnlCusparseCSRMatrix< Real, Device, Index > :: draw( std::ostream& str,
                                                          const String& format,
                                                          tnlCSRMatrix< Real, Device, Index >* csrMatrix,
                                                          int verbose )
{
   Assert( false, );
   return false;
}

template< typename Real, typename Device, typename Index >
void tnlCusparseCSRMatrix< Real, Device, Index > :: printOut( std::ostream& stream,
                                                              const String& format,
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
