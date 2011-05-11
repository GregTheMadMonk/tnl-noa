/***************************************************************************
                          tnlMatrix.h  -  description
                             -------------------
    begin                : 2007/07/23
    copyright            : (C) 2007 by Tomas Oberhuber
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

#ifndef tnlMatrixH
#define tnlMatrixH

#include <ostream>
#include <iomanip>
#include <string.h>
#include <core/tnlObject.h>
#include <core/tnlString.h>
#include <core/tnlList.h>
#include <core/tnlFile.h>
#include <core/tnlLongVector.h>

using namespace std;

class tnlMatrixClass
{
   private:

   tnlMatrixClass() {};

   public:
   static const tnlString main;
   static const tnlString petsc;
};

template< typename Real, tnlDevice Device = tnlHost, typename Index = int >
class tnlMatrix : public tnlObject
{
   public:

   tnlMatrix( const tnlString& name );

   //! Matrix class tells what implementation of matrix we want.
   /*! Matrix class can be main, PETSC, CUDA etc.
    */
   virtual const tnlString& getMatrixClass() const = 0;

   //! Returns the number of rows resp. columns.
   virtual Index getSize() const { return size; };

   //! Use this to change the number of the rows and columns.
   virtual bool setSize( int new_size ) = 0;

   //! Allocates the arrays for the non-zero elements
   virtual bool setNonzeroElements( int n ) = 0;

   virtual Index getNonzeroElementsInRow( const Index& row ) const;

   //! Returns the number of the nonzero elements.
   virtual Index getNonzeroElements() const = 0;

   virtual Index getArtificialZeroElements() const;

   bool setRowsReordering( const tnlLongVector< Index, Device, Index >& reorderingPermutation );

   virtual Real getElement( Index row, Index column ) const = 0;

   //! Setting given element
   /*! Returns false if fails to allocate the new element
    */
   virtual bool setElement( Index row, Index column, const Real& v ) = 0;

   virtual bool addToElement( Index row, Index column, const Real& v ) = 0;
   
   virtual Real rowProduct( const Index row,
                            const tnlLongVector< Real, Device, Index >& vec ) const = 0;
   
   virtual void vectorProduct( const tnlLongVector< Real, Device, Index >& vec,
                               tnlLongVector< Real, Device, Index >& result ) const = 0;

   virtual bool performSORIteration( const Real& omega,
                                     const tnlLongVector< Real, Device, Index >& b,
                                     tnlLongVector< Real, Device, Index >& x,
                                     Index firstRow,
                                     Index lastRow ) const;

   virtual Real getRowL1Norm( Index row ) const = 0;

   virtual void multiplyRow( Index row, const Real& value ) = 0;

   bool operator == ( const tnlMatrix< Real, Device, Index >& m ) const;

   bool operator != ( const tnlMatrix< Real, Device, Index >& m ) const;

   //! Method for saving the matrix to a file as a binary data
   bool save( tnlFile& file ) const;

   //! Method for restoring the matrix from a file
   bool load( tnlFile& file );

   bool save( const tnlString& fileName ) const;

   bool load( const tnlString& fileName );

   /*!
    * Computes permutation of the rows such that the rows would be
    * ordered decreasingly by the number of the non-zero elements.
    */
   bool reorderDecreasingly( const tnlLongVector< Index, Device, Index >& permutation );

   virtual bool read( istream& str,
		                int verbose = 0 );

   virtual bool draw( ostream& str,
		                const tnlString& format,
		                int verbose = 0 );

   virtual void printOut( ostream& stream, const Index lines = 0 ) const {};

   virtual ~tnlMatrix()
   {};

   protected:

   bool checkMtxHeader( const tnlString& header,
		                  bool& symmetric );

   Index size;

   tnlLongVector< Index, Device, Index > rowsReorderingPermutation;
};

template< typename Real, tnlDevice Device, typename Index >
ostream& operator << ( ostream& o_str, const tnlMatrix< Real, Device, Index >& A );

template< typename Real, tnlDevice Device, typename Index >
tnlMatrix< Real, Device, Index > :: tnlMatrix( const tnlString& name )
: tnlObject( name ),
  rowsReorderingPermutation( "tnlMatrix::rowsReorderingPermutation" )
{
};

template< typename Real, tnlDevice Device, typename Index >
Index tnlMatrix< Real, Device, Index > :: getArtificialZeroElements() const
{
   return 0;
};

template< typename Real, tnlDevice Device, typename Index >
Index tnlMatrix< Real, Device, Index > :: getNonzeroElementsInRow( const Index& row ) const
{
   tnlAssert( false, "not implemented yet" );
   /*
    * TODO: this method should be abstract
    */
}

template< typename Real, tnlDevice Device, typename Index >
bool tnlMatrix< Real, Device, Index > :: setRowsReordering( const tnlLongVector< Index, Device, Index >& reorderingPermutation )
{
   if( ! rowsReorderingPermutation. setSize( reorderingPermutation. getSize() ) )
      return false;
   rowsReorderingPermutation = reorderingPermutation;
   return true;
};

template< typename Real, tnlDevice Device, typename Index >
bool tnlMatrix< Real, Device, Index > :: performSORIteration( const Real& omega,
                                                              const tnlLongVector< Real, Device, Index >& b,
                                                              tnlLongVector< Real, Device, Index >& x,
                                                              Index firstRow,
                                                              Index lastRow ) const
{
   return false;
}

template< typename Real, tnlDevice Device, typename Index >
bool tnlMatrix< Real, Device, Index > :: operator == ( const tnlMatrix< Real, Device, Index >& m ) const
{
   if( this -> getSize() != m. getSize() )
      return false;
   const Index size = this -> getSize();
   for( Index i = 0; i < size; i ++ )
      for( Index j = 0; j < size; j ++ )
         if( this -> getElement( i, j ) != m. getElement( i, j ) )
         {
        	 cerr << "Matrices differ at element ( " << i << ", " << j << " )." << endl;
        	 cerr << this -> getName() << " = " << this -> getElement( i, j ) << endl;
        	 cerr << m. getName() << " = " << m. getElement( i, j ) << endl;
             return false;
         }
   return true;
}

template< typename Real, tnlDevice Device, typename Index >
bool tnlMatrix< Real, Device, Index > :: operator != ( const tnlMatrix< Real, Device, Index >& m ) const
{
   return ! ( ( *this ) == m );
}

template< typename Real, tnlDevice Device, typename Index >
bool tnlMatrix< Real, Device, Index > :: save( tnlFile& file ) const
{
   if( ! tnlObject :: save( file ) ) return false;
   if( ! file. write( &size, 1 ) )
      return false;
   return true;
}

template< typename Real, tnlDevice Device, typename Index >
bool tnlMatrix< Real, Device, Index > :: load( tnlFile& file )
{
   if( ! tnlObject :: load( file ) ) return false;
   if( ! file. read( &size, 1 ) )
      return false;
   return true;
}

template< typename Real, tnlDevice Device, typename Index >
bool tnlMatrix< Real, Device, Index > :: save( const tnlString& fileName ) const
{
   return tnlObject :: save( fileName );
}

template< typename Real, tnlDevice Device, typename Index >
bool tnlMatrix< Real, Device, Index > :: load( const tnlString& fileName )
{
   return tnlObject :: load( fileName );
}

template< typename Real, tnlDevice Device, typename Index >
bool tnlMatrix< Real, Device, Index > :: checkMtxHeader( const tnlString& header,
		                                   bool& symmetric )
{
	tnlList< tnlString > parsed_line;
    header. parse( parsed_line );
    if( parsed_line. getSize() < 5 )
       return false;
    if( parsed_line[ 0 ] != "%%MatrixMarket" )
       return false;
    if( parsed_line[ 1 ] != "matrix" )
    {
       cerr << "Error: 'matrix' expected in the header line (" << header << ")." << endl;
       return false;
    }
    if( parsed_line[ 2 ] != "coordinates" &&
        parsed_line[ 2 ] != "coordinate" )
    {
       cerr << "Error: Only 'coordinates' format is supported now, not " << parsed_line[ 2 ] << "." << endl;
       return false;
    }
    if( parsed_line[ 3 ] != "real" )
    {
       cerr << "Error: Only 'real' matrices are supported, not " << parsed_line[ 3 ] << "." << endl;
       return false;
    }
    if( parsed_line[ 4 ] != "general" )
    {
    	if( parsed_line[ 4 ] == "symmetric" )
    		symmetric = true;
    	else
    	{
    		cerr << "Error: Only 'general' matrices are supported, not " << parsed_line[ 4 ] << "." << endl;
    		return false;
    	}
    }
    return true;
}


template< typename Real, tnlDevice Device, typename Index >
bool tnlMatrix< Real, Device, Index > :: read( istream& file,
                                               int verbose )
{
   tnlString line;
   bool dimensions_line( false ), format_ok( false );
   tnlList< tnlString > parsed_line;
   Index non_zero_elements( 0 );
   Index parsed_elements( 0 );
   Index size( 0 );
   bool symmetric( false );
   while( line. getLine( file ) )
   {
      if( ! format_ok )
      {
         format_ok = checkMtxHeader( line, symmetric );
         if( format_ok && verbose )
         {
        	 if( symmetric )
        		 cout << "The matrix is SYMMETRIC." << endl;
         }
         continue;
      }
      if( line[ 0 ] == '%' ) continue;
      if( ! format_ok )
      {
         cerr << "Uknown format of the file. We expect line like this:" << endl;
         cerr << "%%MatrixMarket matrix coordinate real general" << endl;
         return false;
      }

      if( ! dimensions_line )
      {
         parsed_line. EraseAll();
        line. parse( parsed_line );
         if( parsed_line. getSize() != 3 )
         {
           cerr << "Wrong number of parameters in the matrix header." << endl;
           return false;
         }
         Index M = atoi( parsed_line[ 0 ]. getString() );
         Index N = atoi( parsed_line[ 1 ]. getString() );
         Index L = atoi( parsed_line[ 2 ]. getString() );
         if( symmetric )
        	 L = 2 * L - M;
         cout << "Matrix size:       " << setw( 9 ) << right << M << endl;
         cout << "Non-zero elements: " << setw( 9 ) << right << L << endl;

         if( M <= 0 || N <= 0 || L <= 0 )
         {
           cerr << "Wrong parameters in the matrix header." << endl;
           return false;
         }
         if( M  != N )
         {
           cerr << "There is not square matrix in the file." << endl;
           return false;
         }
         if( ! this -> setSize( M ) ||
             ! this -> setNonzeroElements( L ) )
         {
            cerr << "Not enough memory to allocate the sparse or the full matrix for testing." << endl;
            return false;
         }

         dimensions_line = true;
         non_zero_elements = L;
         size = M;
         continue;
      }
      if( parsed_line. getSize() != 3 )
      {
         cerr << "Wrong number of parameters in the matrix row at line:" << line << endl;
         return false;
      }
      parsed_line. EraseAll();
      line. parse( parsed_line );
      Index I = atoi( parsed_line[ 0 ]. getString() );
      Index J = atoi( parsed_line[ 1 ]. getString() );
      double A = atof( parsed_line[ 2 ]. getString() );
      parsed_elements ++;
      if( verbose )
         cout << "Parsed elements:   " << setw( 9 ) << right << parsed_elements << "\r" << flush;
      this -> setElement( I - 1, J - 1, A );
      if( symmetric && I != J )
    	  this -> setElement( J - 1, I - 1, A );
   }
   return true;
}

template< typename Real, tnlDevice Device, typename Index >
bool tnlMatrix< Real, Device, Index > :: reorderDecreasingly( const tnlLongVector< Index, Device, Index >& permutation )
{
   /*
    * We use bucketsort to sort the rows by the number of the non-zero elements.
    */
   if( ! permutation. setSize( this -> getSize() ) )
      return false;
   permutation. setValue( 0 );

   /*
    * The permutation vector is now used to compute the buckets
    */
   for( Index i = 0; i < this -> getSize(); i ++ )
      permutation[ this -> getNonzeroElementsInRow( i ) ] ++;

   tnlLongVector< Index, tnlHost, Index > buckets( "tnlMatrix::reorderDecreasingly:buckets" );
   buckets. setValue( 0 );

   buckets[ 0 ] = 0;
   for( Index i = 1; i < this -> getSize; i ++ )
      buckets[ i ] = buckets[ i - 1 ] + permutation[ i ];

   for( Index i = 1; i < this -> getSize(); i ++ )
      permutation[ buckets[ this -> getNonzeroElementsInRow( i ) ] ++ ] = i;
}

template< typename Real, tnlDevice Device, typename Index >
bool tnlMatrix< Real, Device, Index > :: draw( ostream& str,
		                                         const tnlString& format,
		                                         int verbose )
{
	if( format == "gnuplot" )
	{
		for( Index row = 0; row < getSize(); row ++ )
		{
			for( Index column = 0; column < getSize(); column ++ )
			{
				Real elementValue = getElement( row, column );
				if(  elementValue != ( Real ) 0.0 )
					str << column << " " << getSize() - row << " " << elementValue << endl;
			}
			if( verbose )
				cout << "Drawing the row " << row << "      \r" << flush;
		}
		if( verbose )
			cout << endl;
		return true;
	}
	cerr << "Uknown format " << format << " drawing the matrix." << endl;
	return false;
}

//! Operator <<
template< typename Real, tnlDevice Device, typename Index >
ostream& operator << ( ostream& o_str,
		                 const tnlMatrix< Real, Device, Index >& A )
{
   Index size = A. getSize();
   o_str << endl;
   for( Index i = 0; i < size; i ++ )
   {
      for( Index j = 0; j < size; j ++ )
      {
         const Real& v = A. getElement( i, j );
         if( v == 0.0 ) o_str << setw( 12 ) << ".";
         else o_str << setprecision( 6 ) << setw( 12 ) << v;
      }
      o_str << endl;
   }
   return o_str;
};

#endif
