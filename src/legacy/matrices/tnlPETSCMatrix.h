/***************************************************************************
                          tnlPETSCMatrix.h  -  description
                             -------------------
    begin                : 2008/05/09
    copyright            : (C) 2008 by Tomas Oberhuber
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

#ifndef tnlPETSCMatrixH
#define tnlPETSCMatrixH

#ifdef HAVE_PETSC
#include <petscmat.h>
#include <petscvec.h>
#else
struct Mat;
#endif

#include <matrices/tnlMatrix.h>

#ifdef UNDEF

//! Wrapper for the PETSC matrices
template< typename T > class tnlPETSCMatrix : public tnlMatrix< T >
{

#ifdef HAVE_PETSC
   Mat matrix;
#endif

   int size;

   public:

   tnlPETSCMatrix(){};

   tnlPETSCMatrix( const int _size,
                 const int row_size )
   : tnlMatrix< T >( "tnlPETSCMatrix" ),
     size( _size )
   {
#ifdef HAVE_PETSC
      Mat matrix2;
      MatCreateSeqAIJ( MPI_COMM_SELF, size, size, row_size, PETSC_NULL, &matrix );
#else
      cerr << "Missing support for PETSC at the file " << __FILE__ << " line " << __LINE__ << endl;
#endif
   };

   tnlString getType() const
   {
      T t;
      return tnlString( "tnlPETSCMatrix< " ) + tnlString( GetParameterType( t ) ) + tnlString( " >" );
   };

   const tnlString& getMatrixClass() const
   {
      return tnlMatrixClass :: petsc;
   };
   
   void GetData( Mat& _matrix )
   {
#ifdef HAVE_PETSC
      _matrix = matrix;
#else
      cerr << "Missing support for PETSC at the file " << __FILE__ << " line " << __LINE__ << endl;
#endif
   }
   
   int getSize() const
   {
      return size;
   };

   bool setSize( int n ) {
      assert( 0 );
   };

   T getElement( const int row,
                 const int col ) const
   {
      abort(); // Not implemented yet.
   };

   bool setNonzeroElements( int n ) {
      assert( 0 );
   };

   int getNonzeroElements() const {
      assert( 0 );
   };

   bool setElement( const int row,
                    const int col,
                    const T& v )
   {
#ifdef HAVE_PETSC
      MatSetValue( matrix, row, col, v, INSERT_VALUES );
      return true;
#else
      cerr << "Missing support for PETSC at the file " << __FILE__ << " line " << __LINE__ << endl;
      return false;
#endif
   };

   bool addToElement( const int row,
                      const int col,
                      const T& v )
   {
#ifdef HAVE_PETSC
      MatSetValue( matrix, row, col, v, ADD_VALUES );
      return true;
#else
      cerr << "Missing support for PETSC at the file " << __FILE__ << " line " << __LINE__ << endl;
      return false;
#endif
   };
   
   T rowProduct( const int row,
                 const tnlVector< T >& vec ) const {};
   
   void vectorProduct( const tnlVector< T >&* vec,
                       tnlVector< T >&* result ) const
   {
#ifdef HAVE_PETSC
      Vec petsc_vec, petsc_res;
      VecCreateSeqWithArray( MPI_COMM_SELF, size, vec, &petsc_vec );
      VecCreateSeqWithArray( MPI_COMM_SELF, size, result, &petsc_res );
      
      MatAssemblyBegin( matrix, MAT_FINAL_ASSEMBLY );
      MatAssemblyEnd( matrix, MAT_FINAL_ASSEMBLY );
      
      MatMult( matrix, petsc_vec, petsc_res );
      
      VecDestroy( petsc_vec );
      VecDestroy( petsc_res );
#else
      cerr << "Missing support for PETSC at the file " << __FILE__ << " line " << __LINE__ << endl;
#endif
      
   };
   
   T getRowL1Norm( int row ) const {};

   void multiplyRow( int row, const T& value ) {};

   ~tnlPETSCMatrix()
   {
#ifdef HAVE_PETSC
      MatDestroy( matrix );
#else
      cerr << "Missing support for PETSC at the file " << __FILE__ << " line " << __LINE__ << endl;
#endif
   };

};

#endif
#endif //UNDEF