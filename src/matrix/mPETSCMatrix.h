/***************************************************************************
                          mPETSCMatrix.h  -  description
                             -------------------
    begin                : 2008/05/09
    copyright            : (C) 2008 by Tomá¹ Oberhuber
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

#ifndef mPETSCMatrixH
#define mPETSCMatrixH

#ifdef HAVE_PETSC
#include <petscmat.h>
#include <petscvec.h>
#else
struct Mat;
#endif

#include <matrix/mBaseMatrix.h>

//! Wrapper for the PETSC matrices
template< typename T > class mPETSCMatrix : public mBaseMatrix< T >
{

#ifdef HAVE_PETSC
   Mat matrix;
#endif

   long int size;

   public:

   mPETSCMatrix(){};

   mPETSCMatrix( const long int _size,
                 const long int row_size )
   : size( _size )
   {
#ifdef HAVE_PETSC
      Mat matrix2;
      MatCreateSeqAIJ( MPI_COMM_SELF, size, size, row_size, PETSC_NULL, &matrix );
#else
      cerr << "Missing support for PETSC at the file " << __FILE__ << " line " << __LINE__ << endl;
#endif
   };

   tnlString GetType() const
   {
      T t;
      return tnlString( "mPETSCMatrix< " ) + tnlString( GetParameterType( t ) ) + tnlString( " >" );
   };

   const tnlString& GetMatrixClass() const
   {
      return mMatrixClass :: petsc;
   };
   
   void GetData( Mat& _matrix )
   {
#ifdef HAVE_PETSC
      _matrix = matrix;
#else
      cerr << "Missing support for PETSC at the file " << __FILE__ << " line " << __LINE__ << endl;
#endif
   }
   
   long int GetSize() const
   {
      return size;
   };

   T GetElement( const long int row,
                 const long int col ) const
   {
      abort(); // Not implemented yet.
   };

   bool SetElement( const long int row,
                    const long int col,
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

   bool AddToElement( const long int row,
                      const long int col,
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
   
   T RowProduct( const long int row, const T* vec ) const {};
   
   void VectorProduct( const T* vec, T* result ) const   
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
   
   T GetRowL1Norm( long int row ) const {};

   void MultiplyRow( long int row, const T& value ) {};

   ~mPETSCMatrix()
   {
#ifdef HAVE_PETSC
      MatDestroy( matrix );
#else
      cerr << "Missing support for PETSC at the file " << __FILE__ << " line " << __LINE__ << endl;
#endif
   };

};

#endif
