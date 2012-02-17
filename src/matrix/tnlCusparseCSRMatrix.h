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
#include <core/tnlLongVectorHost.h>
#include <core/tnlAssert.h>
#include <core/mfuncs.h>
#include <matrix/tnlCSRMatrix.h>
#include <debug/tnlDebug.h>

using namespace std;

//! Wrapper for Cusparse CSR matrix
/*!
 */
template< typename Real, tnlDevice Device = tnlHost, typename Index = int  >
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

   //! Allocate memory for the nonzero elements.
   bool setNonzeroElements( Index elements );

   void reset();

   Index getNonzeroElements() const;

   Index getArtificialZeroElements() const;

   bool setElement( Index row,
                    Index colum,
                    const Real& value )
   { abort(); };

   bool addToElement( Index row,
                      Index column,
                      const Real& value )
   { abort(); };


   bool copyFrom( const tnlCSRMatrix< Real, tnlHost, Index >& csr_matrix );

   template< tnlDevice Device2 >
   bool copyFrom( const tnlRgCSRMatrix< Real, Device2, Index >& rgCSRMatrix );

   Real getElement( Index row,
                    Index column ) const;

   Real rowProduct( Index row,
                    const tnlLongVector< Real, Device, Index >& vector ) const;

   void vectorProduct( const tnlLongVector< Real, Device, Index >& x,
                       tnlLongVector< Real, Device, Index >& b ) const;

   Real getRowL1Norm( Index row ) const
   { abort(); };

   void multiplyRow( Index row, const Real& value )
   { abort(); };

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

   tnlLongVector< Real, Device, Index > nonzero_elements;

   tnlLongVector< Index, Device, Index > columns;

   tnlLongVector< Index, Device, Index > row_offsets;

};

template< typename Real, tnlDevice Device, typename Index >
tnlCusparseCSRMatrix< Real, Device, Index > :: tnlCusparseCSRMatrix( const tnlString& name )
   : tnlMatrix< Real, Device, Index >( name ),
     nonzero_elements( name + " : nonzero-elements" ),
     columns( name + " : columns" ),
     row_offsets( name + " : row_offsets" )
{
   cusparseCreate( &cusparseHandle );
   cusparseCreateMatDescr( &cusparseMatDescr );
   cusparseSetMatType( cusparseMatDescr, CUSPARSE_MATRIX_TYPE_GENERAL );
   cusparseSetMatIndexBase( cusparseMatDescr, CUSPARSE_INDEX_BASE_ZERO );
};

template< typename Real, tnlDevice Device, typename Index >
const tnlString& tnlCusparseCSRMatrix< Real, Device, Index > :: getMatrixClass() const
{
   return tnlMatrixClass :: cusparse;
};

template< typename Real, tnlDevice Device, typename Index >
tnlString tnlCusparseCSRMatrix< Real, Device, Index > :: getType() const
{
   return tnlString( "tnlCusparseCSRMatrix< ") +
          tnlString( GetParameterType( Real( 0.0 ) ) ) +
          tnlString( ", " ) +
          getDeviceType( Device ) +
          tnlString( " >" );
};

template< typename Real, tnlDevice Device, typename Index >
bool tnlCusparseCSRMatrix< Real, Device, Index > :: setSize( Index new_size )
{
   this -> size = new_size;
   if( ! row_offsets. setSize( this -> size + 1 ) )
      return false;
   row_offsets. setValue( 0 );
   last_nonzero_element = 0;
   return true;
};

template< typename Real, tnlDevice Device, typename Index >
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
   last_nonzero_element = 0;
   return true;
}

template< typename Real, tnlDevice Device, typename Index >
void tnlCusparseCSRMatrix< Real, Device, Index > :: reset()
{
   nonzero_elements. reset();
   columns. reset();
   row_offsets. reset();
   last_nonzero_element = 0;
}

template< typename Real, tnlDevice Device, typename Index >
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

template< typename Real, tnlDevice Device, typename Index >
Index tnlCusparseCSRMatrix< Real, Device, Index > :: getNonzeroElements() const
{
   return nonzero_elements. getSize();
}

template< typename Real, tnlDevice Device, typename Index >
Index tnlCusparseCSRMatrix< Real, Device, Index > :: ~tnlCusparseCSRMatrix()
{
   cusparseDestroyMatDescr( cusparseMatDescr );
   cusparseDestroy( cusparseHandle );
}









#endif  /* TNLSPMVBENCHMARKCUSPARSEMATRIX_H_ */
