/***************************************************************************
                          tnlMatrix.h  -  description
                             -------------------
    begin                : 2007/07/23
    copyright            : (C) 2007 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef tnlMatrixH
#define tnlMatrixH

#include <ostream>
#include <iomanip>
#include <string.h>
#include <TNL/Object.h>
#include <TNL/String.h>
#include <TNL/List.h>
#include <TNL/File.h>
#include <TNL/Vectors/Vector.h>
#include <TNL/debug/tnlDebug.h>

using namespace std;

class tnlMatrixClass
{
   private:

   tnlMatrixClass() {};

   public:
   static const String main;
   static const String petsc;
   static const String cusparse;
};

template< typename Real, typename device, typename Index > class tnlCSRMatrix;

template< typename Real, typename Device = tnlHost, typename Index = int >
class tnlMatrix : public Object
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;


   tnlMatrix( const String& name );

   //! Matrix class tells what implementation of matrix we want.
   /*! Matrix class can be main, PETSC, CUDA etc.
    */
   virtual const String& getMatrixClass() const = 0;

   //! Returns the number of rows resp. columns.
   virtual Index getSize() const { return size; };

   //! Use this to change the number of the rows and columns.
   virtual bool setSize( Index new_size ) = 0;

   //! Allocates the arrays for the non-zero elements
   virtual bool setNonzeroElements( Index n ) = 0;

   virtual void reset() = 0;

   virtual Index getNonzeroElementsInRow( const Index& row ) const;

   //! Returns the number of the nonzero elements.
   virtual Index getNonzeroElements() const = 0;

   virtual Index getArtificialZeroElements() const;

   //bool setRowsReordering( const Vector< Index, Device, Index >& reorderingPermutation );

   virtual Real getElement( Index row, Index column ) const = 0;

   //! Setting given element
   /*! Returns false if fails to allocate the new element
    */
   virtual bool setElement( Index row, Index column, const Real& v ) = 0;

   virtual bool addToElement( Index row, Index column, const Real& v ) = 0;
 
   virtual Real rowProduct( const Index row,
                            const Vector< Real, Device, Index >& vec ) const = 0;
 
   template< typename Vector1, typename Vector2 >
   void vectorProduct( const Vector1& vec,
                       Vector2& result ) const{}

   virtual bool performSORIteration( const Real& omega,
                                     const Vector< Real, Device, Index >& b,
                                     Vector< Real, Device, Index >& x,
                                     Index firstRow,
                                     Index lastRow ) const;

   virtual Real getRowL1Norm( Index row ) const = 0;

   virtual void multiplyRow( Index row, const Real& value ) = 0;

   bool operator == ( const tnlMatrix< Real, Device, Index >& m ) const;

   bool operator != ( const tnlMatrix< Real, Device, Index >& m ) const;

   /*!***
    * This method is the same as operator == but it can work in verbose mode
    * which is useful when comparing large matrices.
    */
   bool compare( const tnlMatrix< Real, Device, Index >& m, bool verbose = true ) const;

   //! Method for saving the matrix to a file as a binary data
   bool save( File& file ) const;

   //! Method for restoring the matrix from a file
   bool load( File& file );

   bool save( const String& fileName ) const;

   bool load( const String& fileName );

   template< typename Real2 >
   tnlMatrix< Real, Device, Index >& operator = ( const tnlMatrix< Real2, Device, Index >& matrix );

   /*!
    * Computes permutation of the rows such that the rows would be
    * ordered decreasingly by the number of the non-zero elements.
    */
   bool sortRowsDecreasingly( Vector< Index, Device, Index >& permutation );

   virtual bool read( std::istream& str,
		                int verbose = 0 );

   /****
    * If we draw sparse matrix it is much faster if we now positions of the non-zero elements.
    * They are best accessible from the CSR format. Therefore we may pass pointer to tnlCSRMatrix.
    */
   virtual bool draw( std::ostream& str,
		                const String& format,
		                tnlCSRMatrix< Real, Device, Index >* csrMatrix = 0,
		                int verbose = 0 );

   virtual void printOut( std::ostream& stream,
                          const String& format = String( "" ),
                          const Index lines = 0 ) const {};

   virtual ~tnlMatrix()
   {};

   protected:

   bool checkMtxHeader( const String& header,
		                  bool& symmetric );

   void writePostscriptHeader( std::ostream& str,
                               const int elementSize ) const;

   virtual void writePostscriptBody( std::ostream& str,
                                     const int elementSize,
                                     bool verbose ) const;

   Index size;
};

template< typename Real, typename Device, typename Index >
ostream& operator << ( std::ostream& o_str, const tnlMatrix< Real, Device, Index >& A );

template< typename Real, typename Device, typename Index >
tnlMatrix< Real, Device, Index > :: tnlMatrix( const String& name )
: Object( name )
{
};

template< typename Real, typename Device, typename Index >
Index tnlMatrix< Real, Device, Index > :: getArtificialZeroElements() const
{
   return 0;
};

template< typename Real, typename Device, typename Index >
Index tnlMatrix< Real, Device, Index > :: getNonzeroElementsInRow( const Index& row ) const
{
   Assert( false, std::cerr << "not implemented yet." );
   /*
    * TODO: this method should be abstract
    */
   abort();
}

template< typename Real, typename Device, typename Index >
bool tnlMatrix< Real, Device, Index > :: performSORIteration( const Real& omega,
                                                              const Vector< Real, Device, Index >& b,
                                                              Vector< Real, Device, Index >& x,
                                                              Index firstRow,
                                                              Index lastRow ) const
{
   return false;
}

template< typename Real, typename Device, typename Index >
bool tnlMatrix< Real, Device, Index > :: operator == ( const tnlMatrix< Real, Device, Index >& m ) const
{
   return compare( m, false );
}

template< typename Real, typename Device, typename Index >
bool tnlMatrix< Real, Device, Index > :: compare( const tnlMatrix< Real, Device, Index >& m, bool verbose ) const
{
   if( this->getSize() != m. getSize() )
      return false;
   const Index size = this->getSize();
   for( Index i = 0; i < size; i ++ )
      for( Index j = 0; j < size; j ++ )
      {
         if( verbose )
           std::cout << "Comparing: " << i << " / " << size << "\r";
         if( this->getElement( i, j ) != m. getElement( i, j ) )
             return false;
      }
   return true;
}

template< typename Real, typename Device, typename Index >
bool tnlMatrix< Real, Device, Index > :: operator != ( const tnlMatrix< Real, Device, Index >& m ) const
{
   return ! ( ( *this ) == m );
}

template< typename Real, typename Device, typename Index >
bool tnlMatrix< Real, Device, Index > :: save( File& file ) const
{
   if( ! Object :: save( file ) ) return false;
#ifdef HAVE_NOT_CXX11
   if( ! file. write< const Index, tnlHost >( &size ) )
#else
   if( ! file. write( &size ) )
#endif
      return false;
   return true;
}

template< typename Real, typename Device, typename Index >
bool tnlMatrix< Real, Device, Index > :: load( File& file )
{
   if( ! Object :: load( file ) ) return false;
#ifdef HAVE_NOT_CXX11
   if( ! file. read< Index, tnlHost >( &size ) )
#else
   if( ! file. read( &size ) )
#endif
      return false;
   return true;
}

template< typename Real, typename Device, typename Index >
bool tnlMatrix< Real, Device, Index > :: save( const String& fileName ) const
{
   return Object :: save( fileName );
}

template< typename Real, typename Device, typename Index >
bool tnlMatrix< Real, Device, Index > :: load( const String& fileName )
{
   return Object :: load( fileName );
}

template< typename Real, typename Device, typename Index >
   template< typename Real2 >
tnlMatrix< Real, Device, Index >& tnlMatrix< Real, Device, Index > ::  operator = ( const tnlMatrix< Real2, Device, Index >& matrix )
{
   this->size = matrix. getSize();
   /*if( ! rowsReorderingPermutation. setSize( matrix. rowsReorderingPermutation. getSize() ) )
   {
      std::cerr << "I am not able to allocat the row permutation vector for the new matrix." << std::endl;
      abort();
   }
   rowsReorderingPermutation = matrix. rowsReorderingPermutation;*/
   return * this;
}

template< typename Real, typename Device, typename Index >
bool tnlMatrix< Real, Device, Index > :: checkMtxHeader( const String& header,
		                                                   bool& symmetric )
{
	List< String > parsed_line;
    header. parse( parsed_line );
    if( parsed_line. getSize() < 5 )
       return false;
    if( parsed_line[ 0 ] != "%%MatrixMarket" )
       return false;
    if( parsed_line[ 1 ] != "matrix" )
    {
       std::cerr << "Error: 'matrix' expected in the header line (" << header << ")." << std::endl;
       return false;
    }
    if( parsed_line[ 2 ] != "coordinates" &&
        parsed_line[ 2 ] != "coordinate" )
    {
       std::cerr << "Error: Only 'coordinates' format is supported now, not " << parsed_line[ 2 ] << "." << std::endl;
       return false;
    }
    if( parsed_line[ 3 ] != "real" )
    {
       std::cerr << "Error: Only 'real' matrices are supported, not " << parsed_line[ 3 ] << "." << std::endl;
       return false;
    }
    if( parsed_line[ 4 ] != "general" )
    {
    	if( parsed_line[ 4 ] == "symmetric" )
    		symmetric = true;
    	else
    	{
    		cerr << "Error: Only 'general' matrices are supported, not " << parsed_line[ 4 ] << "." << std::endl;
    		return false;
    	}
    }
    return true;
}


template< typename Real, typename Device, typename Index >
bool tnlMatrix< Real, Device, Index > :: read( std::istream& file,
                                               int verbose )
{
   String line;
   bool dimensions_line( false ), format_ok( false );
   List< String > parsed_line;
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
        		std::cout << "The matrix is SYMMETRIC." << std::endl;
         }
         continue;
      }
      if( line[ 0 ] == '%' ) continue;
      if( ! format_ok )
      {
         std::cerr << "Uknown format of the file. We expect line like this:" << std::endl;
         std::cerr << "%%MatrixMarket matrix coordinate real general" << std::endl;
         return false;
      }

      if( ! dimensions_line )
      {
         parsed_line. EraseAll();
         line. parse( parsed_line );
         if( parsed_line. getSize() != 3 )
         {
           std::cerr << "Wrong number of parameters in the matrix header." << std::endl;
           return false;
         }
         Index M = atoi( parsed_line[ 0 ]. getString() );
         Index N = atoi( parsed_line[ 1 ]. getString() );
         Index L = atoi( parsed_line[ 2 ]. getString() );
         if( symmetric )
        	 L = 2 * L - M;
        std::cout << "Matrix size:       " << std::setw( 9 ) << right << M << std::endl;
        std::cout << "Non-zero elements: " << std::setw( 9 ) << right << L << std::endl;

         if( M <= 0 || N <= 0 || L <= 0 )
         {
           std::cerr << "Wrong parameters in the matrix header." << std::endl;
           return false;
         }
         if( M  != N )
         {
           std::cerr << "There is not square matrix in the file." << std::endl;
           return false;
         }
         if( ! this->setSize( M ) ||
             ! this->setNonzeroElements( L ) )
         {
            std::cerr << "Not enough memory to allocate the sparse or the full matrix for testing." << std::endl;
            return false;
         }

         dimensions_line = true;
         size = M;
         continue;
      }
      if( parsed_line. getSize() != 3 )
      {
         std::cerr << "Wrong number of parameters in the matrix row at line:" << line << std::endl;
         return false;
      }
      parsed_line. EraseAll();
      line. parse( parsed_line );
      Index I = atoi( parsed_line[ 0 ]. getString() );
      Index J = atoi( parsed_line[ 1 ]. getString() );
      Real A = ( Real ) atof( parsed_line[ 2 ]. getString() );
      parsed_elements ++;
      if( verbose )
        std::cout << "Parsed elements:   " << std::setw( 9 ) << right << parsed_elements << "\r" << std::flush;
      this->setElement( I - 1, J - 1, A );
      if( symmetric && I != J )
    	  this->setElement( J - 1, I - 1, A );
   }
   return true;
}

template< typename Real, typename Device, typename Index >
bool tnlMatrix< Real, Device, Index > :: sortRowsDecreasingly( Vector< Index, Device, Index >& permutation )
{
   dbgFunctionName( "tnlMatrix< Real, Device, Index >", "sortRowsDecreasingly" );
   /****
    * We use bucketsort to sort the rows by the number of the non-zero elements.
    */
   const Index matrixSize = tnlMatrix< Real, Device, Index > :: getSize();
   if( ! permutation. setSize( matrixSize + 1 ) )
      return false;
   permutation. setValue( 0 );

   /****
    * The permutation vector is now used to compute the buckets
    */
   for( Index i = 0; i < matrixSize; i ++ )
   {
      Assert( this->getNonzeroElementsInRow( i ) <= matrixSize,
                 std::cerr << "getNonzeroElementsInRow( " << i << " ) = " << getNonzeroElementsInRow( i )
                      << "; matrixSize = " << matrixSize );
      permutation[ this->getNonzeroElementsInRow( i ) ] ++;
   }

   Vector< Index, tnlHost, Index > buckets( "tnlMatrix::reorderRowsDecreasingly:buckets" );
   buckets. setSize( matrixSize + 1 );
   buckets. setValue( 0 );

   buckets[ 0 ] = 0;
   for( Index i = 1; i <= matrixSize; i ++ )
   {
      Assert( matrixSize - i >= 0 && matrixSize - i <= matrixSize, );
      buckets[ i ] = buckets[ i - 1 ] + permutation[ matrixSize - i + 1 ];
   }

   for( Index i = 0; i < matrixSize; i ++ )
   {
      Assert( buckets[ matrixSize - this->getNonzeroElementsInRow( i ) ] <= matrixSize,
               std::cerr << "buckets[ matrixSize - this->getNonzeroElementsInRow( i ) - 1 ] = " << buckets[ matrixSize - this->getNonzeroElementsInRow( i ) - 1 ]
                    << "; matrixSize = " << matrixSize );
      dbgExpr( buckets[ matrixSize - this->getNonzeroElementsInRow( i ) ] );
      permutation[ buckets[ matrixSize - this->getNonzeroElementsInRow( i ) ] ++ ] = i;
   }
   return true;
}

template< typename Real, typename Device, typename Index >
bool tnlMatrix< Real, Device, Index > :: draw( std::ostream& str,
		                                         const String& format,
		                                         tnlCSRMatrix< Real, Device, Index >* csrMatrix,
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
					str << column << " " << getSize() - row << " " << elementValue << std::endl;
			}
			if( verbose )
				cout << "Drawing the row " << row << "      \r" << std::flush;
		}
		if( verbose )
			cout << std::endl;
		return true;
	}
	if( format == "eps" )
	{
	   const int elementSize = 10;
	   this->writePostscriptHeader( str, elementSize );
	   if( csrMatrix )
	      csrMatrix -> writePostscriptBody( str, elementSize, verbose );
	   else
	      this->writePostscriptBody( str, elementSize, verbose );

	   str << "showpage" << std::endl;
      str << "%%EOF" << std::endl;

      if( verbose )
        std::cout << std::endl;
      return true;
	}
	cerr << "Unknown format " << format << " for drawing the matrix." << std::endl;
	return false;
}

template< typename Real, typename Device, typename Index >
void tnlMatrix< Real, Device, Index > :: writePostscriptHeader( std::ostream& str,
                                                                const int elementSize ) const
{
   const int scale = elementSize * this->getSize();
   str << "%!PS-Adobe-2.0 EPSF-2.0" << std::endl;
   str << "%%BoundingBox: 0 0 " << scale << " " << scale << std::endl;
   str << "%%Creator: TNL" << std::endl;
   str << "%%LanguageLevel: 2" << std::endl;
   str << "%%EndComments" << std::endl << std::endl;
   str << "0 " << scale << " translate" << std::endl;
}

template< typename Real, typename Device, typename Index >
void tnlMatrix< Real, Device, Index > :: writePostscriptBody( std::ostream& str,
                                                              const int elementSize,
                                                              bool verbose ) const
{
   const double scale = elementSize * this->getSize();
   double hx = scale / ( double ) this->getSize();
   Index lastRow( 0 ), lastColumn( 0 );
   for( Index row = 0; row < getSize(); row ++ )
   {
      for( Index column = 0; column < getSize(); column ++ )
      {
         Real elementValue = getElement( row, column );
         if(  elementValue != ( Real ) 0.0 )
         {
            str << ( column - lastColumn ) * elementSize
                << " " << -( row - lastRow ) * elementSize
                << " translate newpath 0 0 " << elementSize << " " << elementSize << " rectstroke" << std::endl;
            lastColumn = column;
            lastRow = row;
         }
      }
      if( verbose )
        std::cout << "Drawing the row " << row << "      \r" << std::flush;
   }
}

//! Operator <<
template< typename Real, typename Device, typename Index >
ostream& operator << ( std::ostream& o_str,
		                 const tnlMatrix< Real, Device, Index >& A )
{
   Index size = A. getSize();
   o_str << std::endl;
   for( Index i = 0; i < size; i ++ )
   {
      for( Index j = 0; j < size; j ++ )
      {
         const Real& v = A. getElement( i, j );
         if( v == 0.0 ) o_str << std::setw( 12 ) << ".";
         else o_str << std::setprecision( 6 ) << std::setw( 12 ) << v;
      }
      o_str << std::endl;
   }
   return o_str;
};

#endif
