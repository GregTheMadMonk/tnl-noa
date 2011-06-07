/***************************************************************************
 tnlLongVectorCUDA.h  -  description
 -------------------
 begin                : Dec 27, 2009
 copyright            : (C) 2009 by Tomas Oberhuber
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

#ifndef TNLLONGVECTORCUDA_H_
#define TNLLONGVECTORCUDA_H_


/* When we need to transfer data between the GPU and the CPU we use
 * 5 MB buffer. This size should ensure good performance -- see.
 * http://wiki.accelereyes.com/wiki/index.php/GPU_Memory_Transfer
 * Similar constant is defined also in tnlFile.
 */
const int tnlGPUvsCPUTransferBufferSize = 5 * 2<<20;

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#else
#include <iostream>
using namespace std;
#endif

#include <core/tnlAssert.h>
#include <core/tnlObject.h>
#include <core/param-types.h>
#include <core/mfuncs.h>
#include <core/tnlCudaSupport.h>
#include <core/tnlLongVectorBase.h>
#include <core/low-level/cuda-long-vector-kernels.h>
#include <debug/tnlDebug.h>


template< typename Real, typename Index >
class tnlLongVector< Real, tnlCuda, Index > : public tnlLongVectorBase< Real >
{
   //! We do not allow constructor without parameters.
   tnlLongVector(){};

   /****
    * We do not allow copy constructors as well to avoid having two
    * vectors with the same name.
    */
   tnlLongVector( const tnlLongVector< Real, tnlHost, Index >& v ){};

   tnlLongVector( const tnlLongVector< Real, tnlCuda, Index >& v ){};

   public:

   //! Basic constructor
   tnlLongVector( const tnlString& name, int _size = 0 );

   //! Constructor with another long vector as template
   tnlLongVector( const tnlString& name, const tnlLongVector< Real, tnlHost, Index >& v );

   //! Constructor with another long vector as template
   tnlLongVector( const tnlString& name, const tnlLongVector< Real, tnlCuda, Index >& v );

   //! Use this if you want to insert some data in this vector.
   /*!***
    *  The data will not be deallocated by the destructor.
    *  Once setSize method is called the vector forgets the shared data.
    */
   virtual void setSharedData( Real* _data, const Index _size );

   //! Set size of the vector and allocate necessary amount of the memory.
   bool setSize( Index _size );

   //! Set size of the vector using another vector as a template
   bool setSize( const tnlLongVector< Real, tnlCuda, Index >& v );

   //! Set size of the vector using another vector as a template
   bool setSize( const tnlLongVector< Real, tnlHost, Index >& v );

   //! Set size of the vector using another vector as a template
   bool setLike( const tnlLongVector< Real, tnlHost, Index >& v );

   //! Set size of the vector using another vector as a template
   bool setLike( const tnlLongVector< Real, tnlCuda, Index >& v );

   /*!**
    * Free allocated memory
    */
   void reset();

   void swap( tnlLongVector< Real, tnlCuda, Index >& u );

   //! Returns type of this vector written in a form of C++ template type.
   tnlString getType() const;

   //! In the safe mode all GPU operations are synchronized.
   /*!***
    *  It means that for example we wait for all data transfers to be finished.
    * By default it is turned on.
    */
   void setSafeMode( bool mode );

   //! In the safe mode all GPU operations are synchronized.
   /*!***
    *  It means that for example we wait for all data transfers to be finished.
    *  By default it is turned on.
    */
   bool getSafeMode() const;

   //! Set value of one particular element of the vector.
   /*!***
    * Since this vector resides on the GPU this is extremely slow.
    *  Use this method only for very special purpose like debugging
    *  or if you want to touch only few elements.
    *  In production code it is better to set vector on CPU and then
    *  copy it to the GPU using the overloaded operator =.
    */
   void setElement( Index i, Real d );

   //! Get value of one particular element of the vector.
   /*!***
    * Since this vector resides on the GPU this is extremely slow.
    *  Use this method only for very special purpose like debugging
    *  or if you want to read only few elements.
    *  In production code it is better to copy the vector first from the
    *  GPU to CPU by overloaded operator = and then read the results.
    */
   Real getElement( Index i ) const;

   //! This operator CAN NOT be used.
   /*!***
    * We can not return reference pointing to a memory space on the CUDA device.
    * This operator therefore does not makes sense. However, for the completeness
    * of the interface, it must be implemented. If a program reaches this operator,
    * the abort function is raised and explaining error message is written on the console.
    */
   Real& operator[] ( Index i );

   //! This operator CAN NOT be used.
   /*!***
    * We can not return reference pointing to a memory space on the CUDA device.
    * This operator therefore does not makes sense. However, for the completeness
    * of the interface, it must be implemented. If a program reaches this operator,
    * the abort function is raised and explaining error message is written on the console.
    */
   const Real& operator[] ( Index i ) const;

   bool operator == ( const tnlLongVector< Real, tnlCuda, Index >& long_vector ) const;

   bool operator != ( const tnlLongVector< Real, tnlCuda, Index >& long_vector ) const;

   bool operator == ( const tnlLongVector< Real, tnlHost, Index >& long_vector ) const;

   bool operator != ( const tnlLongVector< Real, tnlHost, Index >& long_vector ) const;

   tnlLongVector< Real, tnlCuda, Index >& operator = ( const tnlLongVector< Real, tnlCuda, Index >& cuda_vector );

   tnlLongVector< Real, tnlCuda, Index >& operator = ( const tnlLongVector< Real, tnlHost, Index >& long_vector );

   template< typename Real2, typename Index2 >
   tnlLongVector< Real, tnlCuda, Index >& operator = ( const tnlLongVector< Real2, tnlCuda, Index2 >& long_vector );

   void setValue( const Real& v );

   //! Method for saving the object to a file as a binary data
   virtual bool save( tnlFile& file ) const;

   //! Method for restoring the object from a file
   virtual bool load( tnlFile& file );

    ~tnlLongVector();

   protected:

   //! In this mode all GPU operations run synchronised.
   /*!***
    * It means that for example we wait for all data transfers to be finished.
    */
   bool safeMode;

};

template< typename Real, typename Index >
bool operator == ( const tnlLongVector< Real, tnlHost, Index >& host_vector,
                   const tnlLongVector< Real, tnlCuda, Index >& cuda_vector );

template< typename Real, typename Index >
bool operator != ( const tnlLongVector< Real, tnlHost, Index >& host_vector,
                   const tnlLongVector< Real, tnlCuda, Index >& cuda_vector );

template< typename Real, typename Index >
ostream& operator << ( ostream& str, const tnlLongVector< Real, tnlCuda, Index >& vec );

/****
 * Here are some Blas style functions. They are not methods
 * because it would put too many restrictions on type Real.
 * In fact, we would like to use tnlLongVector to store objects
 * like edges or triangles in case of meshes. For these objects
 * operations like +, min or max are not defined.
 */

template< typename Real, typename Index >
Real tnlMax( const tnlLongVector< Real, tnlCuda, Index >& v );

template< typename Real, typename Index >
Real tnlMin( const tnlLongVector< Real, tnlCuda, Index >& v );

template< typename Real, typename Index >
Real tnlAbsMax( const tnlLongVector< Real, tnlCuda, Index >& v );

template< typename Real, typename Index >
Real tnlAbsMin( const tnlLongVector< Real, tnlCuda, Index >& v );

template< typename Real, typename Index >
Real tnlLpNorm( const tnlLongVector< Real, tnlCuda, Index >& v, const Real& p );

template< typename Real, typename Index >
Real tnlDifferenceMax( const tnlLongVector< Real, tnlCuda, Index >& u,
                       const tnlLongVector< Real, tnlCuda, Index >& v );

template< typename Real, typename Index >
Real tnlDifferenceMin( const tnlLongVector< Real, tnlCuda, Index >& u,
                       const tnlLongVector< Real, tnlCuda, Index >& v );

template< typename Real, typename Index >
Real tnlDifferenceAbsMax( const tnlLongVector< Real, tnlCuda, Index >& u,
                          const tnlLongVector< Real, tnlCuda, Index >& v );

template< typename Real, typename Index >
Real tnlDifferenceAbsMin( const tnlLongVector< Real, tnlCuda, Index >& u,
                          const tnlLongVector< Real, tnlCuda, Index >& v );

template< typename Real, typename Index >
Real tnlDifferenceLpNorm( const tnlLongVector< Real, tnlCuda, Index >& u,
                          const tnlLongVector< Real, tnlCuda, Index >& v, const Real& p );

template< typename Real, typename Index >
Real tnlDifferenceSum( const tnlLongVector< Real, tnlCuda, Index >& u,
                       const tnlLongVector< Real, tnlCuda, Index >& v );

template< typename Real, typename Index >
Real tnlSum( const tnlLongVector< Real, tnlCuda, Index >& v );

//! Computes u *= aplha
template< typename Real, typename Index >
void tnlScalarMultiplication( const Real& alpha,
                              tnlLongVector< Real, tnlCuda, Index >& u );

//! Computes scalar dot product
template< typename Real, typename Index >
Real tnlSDOT( const tnlLongVector< Real, tnlCuda, Index >& u ,
              const tnlLongVector< Real, tnlCuda, Index >& v );

//! Computes SAXPY operation (Scalar Alpha X Pus Y ).
template< typename Real, typename Index >
void tnlSAXPY( const Real& alpha,
               const tnlLongVector< Real, tnlCuda, Index >& x,
               tnlLongVector< Real, tnlCuda, Index >& y );

//! Computes SAXMY operation (Scalar Alpha X Minus Y ).
/*!**
 * It is not a standard BLAS function but it is useful for GMRES solver.
 */
template< typename Real, typename Index >
void tnlSAXMY( const Real& alpha,
               const tnlLongVector< Real, tnlCuda, Index >& x,
               tnlLongVector< Real, tnlCuda, Index >& y );


#ifdef HAVE_CUDA
template< typename Real, typename Index >
__global__ void tnlLongVectorCUDASetValueKernel( Real* data,
                                                 const Index size,
                                                 const Real v  );
#endif


template< typename Real, typename Index >
tnlLongVector< Real, tnlCuda, Index > :: tnlLongVector( const tnlString& name, int _size )
: tnlLongVectorBase< Real >( name ), safeMode( true )
{
  setSize( _size );
};

template< typename Real, typename Index >
tnlLongVector< Real, tnlCuda, Index > :: tnlLongVector( const tnlString& name,
                                                        const tnlLongVector< Real, tnlHost, Index >& v )
   : tnlLongVectorBase< Real >( name ), safeMode( true )
{
   setSize( v. getSize() );
};

template< typename Real, typename Index >
tnlLongVector< Real, tnlCuda, Index > :: tnlLongVector( const tnlString& name,
                                                        const tnlLongVector< Real, tnlCuda, Index >& v )
   : tnlLongVectorBase< Real >( name ), safeMode( true )
{
   setSize( v. getSize() );
};

template< typename Real, typename Index >
void tnlLongVector< Real, tnlCuda, Index > :: setSharedData( Real* _data, const Index _size )
{
#ifdef HAVE_CUDA

   if( this -> data &&
     ! this -> shared_data ) cudaFree( this -> data );
   this -> data = _data;
   this -> shared_data = true;
   this -> size = _size;
#else
   cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
#endif
};

template< typename Real, typename Index >
bool tnlLongVector< Real, tnlCuda, Index > :: setSize( Index _size )
{
   dbgFunctionName( "tnlLongVectorCUDA", "setSize" );
   tnlAssert( _size >= 0,
            cerr << "You try to set size of tnlLongVector to negative value."
                 << "Vector name: " << this -> getName() << endl
                 << "New size: " << _size << endl );
   /* In the case that we run without active macro tnlAssert
    * we will write at least warning.
    */
   if( _size < 0 )
   {
      cerr << "Negative size " << _size << " was passed to tnlLongVector " << this -> getName() << "." << endl;
      return false;
   }
#ifdef HAVE_CUDA
   dbgCout( "Setting new size to " << _size << " for " << this -> getName() );
   if( this -> size && this -> size == _size && ! this -> shared_data ) return true;
   if( this -> data && ! this -> shared_data )
   {
      dbgCout( "Freeing allocated memory on CUDA device of " << this -> getName() );
      cudaFree( this -> data );
      if( ! checkCUDAError( __FILE__, __LINE__ ) )
       return false;
      this -> data = NULL;
   }
   this -> size = _size;
   this -> shared_data = false;
   if( cudaMalloc( ( void** ) & this -> data,
                   ( size_t ) this -> size * sizeof( Real ) ) != cudaSuccess )
   {
      cerr << "I am not able to allocate new long vector with size "
           << ( double ) this -> size * sizeof( Real ) / 1.0e9 << " GB on CUDA device for "
           << this -> getName() << "." << endl;
      checkCUDAError( __FILE__, __LINE__ );
      this -> data = NULL;
      this -> size = 0;
      return false;
   }
   return true;
#else
   //cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
   return false;
#endif
};

template< typename Real, typename Index >
bool tnlLongVector< Real, tnlCuda, Index > :: setSize( const tnlLongVector< Real, tnlHost, Index >& v )
{
   return setSize( v. getSize() );
};

template< typename Real, typename Index >
bool tnlLongVector< Real, tnlCuda, Index > :: setSize( const tnlLongVector< Real, tnlCuda, Index >& v )
{
   return setSize( v. getSize() );
};

template< typename Real, typename Index >
bool tnlLongVector< Real, tnlCuda, Index > :: setLike( const tnlLongVector< Real, tnlHost, Index >& v )
{
   return setSize( v. getSize() );
};

template< typename Real, typename Index >
bool tnlLongVector< Real, tnlCuda, Index > :: setLike( const tnlLongVector< Real, tnlCuda, Index >& v )
{
   return setSize( v. getSize() );
};

template< typename Real, typename Index >
void tnlLongVector< Real, tnlCuda, Index > :: reset()
{
#ifdef HAVE_CUDA
   dbgFunctionName( "tnlLongVectorCUDA", "reset" );
   if( this -> data && ! this -> shared_data )
   {
      dbgCout( "Freeing allocated memory on CUDA device of " << this -> getName() );
      cudaFree( this -> data );
      if( ! checkCUDAError( __FILE__, __LINE__ ) )
      this -> data = NULL;
   }
   this -> size = 0;
   this -> shared_data = false;
   this -> data = 0;
#endif
};

template< typename Real, typename Index >
void tnlLongVector< Real, tnlCuda, Index > :: swap( tnlLongVector< Real, tnlCuda, Index >& v )
{
   tnlAssert( this -> getSize() > 0, );
   tnlAssert( this -> getSize() == v. getSize(),
              cerr << "You try to swap two long vectors with different sizes." << endl
                   << "The first one is " << this -> getName() << " with size " << this -> getSize()
                   << " while the second one is " << v. getName() << " with size " << v. getSize() << "." );

   std :: swap( this -> data, v. data );
   std :: swap( this -> shared_data, v. shared_data );
};

template< typename Real, typename Index >
tnlString tnlLongVector< Real, tnlCuda, Index > :: getType() const
{
    Real t;
    return tnlString( "tnlLongVector< " ) + tnlString( GetParameterType( t ) ) + tnlString( ", tnlCuda >" );
};

template< typename Real, typename Index >
void tnlLongVector< Real, tnlCuda, Index > :: setSafeMode( bool mode )
{
   safeMode = mode;
}

template< typename Real, typename Index >
bool tnlLongVector< Real, tnlCuda, Index > :: getSafeMode() const
{
   return safeMode;
}

template< typename Real, typename Index >
void tnlLongVector< Real, tnlCuda, Index > :: setElement( Index i, Real d )
{
   tnlAssert( this -> size != 0,
              cerr << "Vector name is " << this -> getName() );
   tnlAssert( i < this -> size,
            cerr << "You try to set non-existing element of the following vector."
                 << "Name: " << this -> getName() << endl
                 << "Size: " << this -> size << endl
                 << "Element number: " << i << endl; );
#ifdef HAVE_CUDA
   if( cudaMemcpy( ( void* ) &( this -> data[ i ] ),
                   ( void* ) &d,
                   sizeof( Real ),
                   cudaMemcpyHostToDevice ) != cudaSuccess )
   {
      cerr << "Transfer of data to the element number " << i 
           << " of the CUDA long vector " << this -> getName()
           << " from the host failed." << endl;
      checkCUDAError( __FILE__, __LINE__ );
   }
#else
   cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
   return;
#endif
};

template< typename Real, typename Index >
Real tnlLongVector< Real, tnlCuda, Index > :: getElement( Index i ) const
{
   tnlAssert( this -> size != 0,
              cerr << "Vector name is " << this -> getName() );
   tnlAssert( i < this -> size,
            cerr << "You try to get non-existing element of the following vector."
                 << "Name: " << this -> getName() << endl
                 << "Size: " << this -> size << endl
                 << "Element number: " << i << endl; );
#ifdef HAVE_CUDA
   Real result;
   if( cudaMemcpy( ( void* ) &result,
                   ( void* )&( this -> data[ i ] ),
                   sizeof( Real ),
                   cudaMemcpyDeviceToHost ) != cudaSuccess )
   {
      cerr << "Transfer of data from the element number " << i 
           << " of the CUDA long vector " << this -> getName()
           << " to the host failed." << endl;
      checkCUDAError( __FILE__, __LINE__ );
   }
   return result;
#else
   cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
   return Real();
#endif
};

template< typename Real, typename Index >
Real& tnlLongVector< Real, tnlCuda, Index > :: operator[] ( Index i )
{
   cerr << "I am sorry. You try to call the following operator: " << endl;
   cerr << endl;
   cerr << "Real& tnlLongVector< Real, tnlCuda, Index > :: operator[] ( Index i )" << endl;
   cerr << endl;
   cerr << "for vector " << this -> getName() << "." << endl;
   cerr << "The call comes from the HOST (CPU) and the long vector is allocated " << endl;
   cerr << "on the CUDA device. Therefore we cannot return reference pointing to the" << endl;
   cerr << "different memory space. You may use the method getElement or setElement" << endl;
   cerr << "which are however very slow. You may also write specialised kernel for your task." << endl;
   abort();
}

template< typename Real, typename Index >
const Real& tnlLongVector< Real, tnlCuda, Index > :: operator[] ( Index i ) const
{
   cerr << "I am sorry. You try to call the following operator: " << endl;
   cerr << endl;
   cerr << "Real& tnlLongVector< Real, tnlCuda, Index > :: operator[] ( Index i )" << endl;
   cerr << endl;
   cerr << "for vector " << this -> getName() << "." << endl;
   cerr << "The call comes from the HOST (CPU) and the long vector is allocated " << endl;
   cerr << "on the CUDA device. Therefore we cannot return reference poointing to the" << endl;
   cerr << "different memory space. You may use the method getElement or setElement" << endl;
   cerr << "which are however very slow. You may also write specialised kernel for your task." << endl;
   abort();
}

template< typename Real, typename Index >
bool tnlLongVector< Real, tnlCuda, Index > :: operator == ( const tnlLongVector< Real, tnlCuda, Index >& cuda_vector ) const
{
   tnlAssert( this -> getSize() == cuda_vector. getSize(),
              cerr << "You try to compare two long vectors with different sizes." << endl
                   << "The first one is " << this -> getName() << " with size " << this -> getSize() 
                   << " while the second one is " << cuda_vector. getName() << " with size " << cuda_vector. getSize() << "." );
#ifdef HAVE_CUDA
   if( tnlCUDALongVectorComparison( this -> getSize(),
                                    this -> getVector(),
                                    cuda_vector. getVector() ) )
      return true;
   return false;
#else
   cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
   return;
#endif
};

template< typename Real, typename Index >
bool tnlLongVector< Real, tnlCuda, Index > :: operator != ( const tnlLongVector< Real, tnlCuda, Index >& long_vector ) const
{
   return ! ( ( *this ) == long_vector );
};

template< typename Real, typename Index >
bool tnlLongVector< Real, tnlCuda, Index > :: operator == ( const tnlLongVector< Real, tnlHost, Index >& host_vector ) const
{
   tnlAssert( this -> getSize() == host_vector. getSize(),
              cerr << "You try to compare two long vectors with different sizes." << endl
                   << "The first one is " << this -> getName() << " with size " << this -> getSize()
                   << " while the second one is " << host_vector. getName() << " with size " << host_vector. getSize() << "." );
#ifdef HAVE_CUDA
   Index host_buffer_size = :: Min( ( Index ) ( tnlGPUvsCPUTransferBufferSize / sizeof( Real ) ),
                                    this -> getSize() );
   Real* host_buffer = new Real[ host_buffer_size ];
   if( ! host_buffer )
   {
      cerr << "I am sorry but I cannot allocate supporting buffer on the host for comparing data between CUDA GPU and CPU." << endl;
      return false;
   }
   Index compared( 0 );
   while( compared < this -> getSize() )
   {
      Index transfer = Min( this -> getSize() - compared, host_buffer_size );
      if( cudaMemcpy( ( void* ) host_buffer,
                      ( void* ) & ( this -> getVector()[ compared ] ),
                      transfer * sizeof( Real ),
                      cudaMemcpyDeviceToHost ) != cudaSuccess )
      {
         cerr << "Transfer of data to the element number of the CUDA long vector " << this -> getName()
              << " from the device failed." << endl;
         checkCUDAError( __FILE__, __LINE__ );
         delete[] host_buffer;
         return false;
      }
      Index buffer_index( 0 );
      while( buffer_index < transfer &&
             host_buffer[ buffer_index ] == host_vector. getElement( compared ) )
      {
         buffer_index ++;
         compared ++;
      }
      if( buffer_index < transfer )
      {
         delete[] host_buffer;
         return false;
      }
   }
   delete[] host_buffer;
   return true;
#else
   cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
   return;
#endif
};

template< typename Real, typename Index >
bool tnlLongVector< Real, tnlCuda, Index > :: operator != ( const tnlLongVector< Real, tnlHost, Index >& long_vector ) const
{
   return ! ( ( *this ) == long_vector );
};

template< typename Real, typename Index >
tnlLongVector< Real, tnlCuda, Index >& tnlLongVector< Real, tnlCuda, Index > :: operator = ( const tnlLongVector< Real, tnlHost, Index >& rhs_vector )
{
#ifdef HAVE_CUDA
   tnlAssert( rhs_vector. getSize() == this -> getSize(),
           cerr << "Source name: " << rhs_vector. getName() << endl
                << "Source size: " << rhs_vector. getSize() << endl
                << "Target name: " << this -> getName() << endl
                << "Target size: " << this -> getSize() << endl );

   if( cudaMemcpy( this -> data,
                   rhs_vector. getVector(),
                   this -> getSize() * sizeof( Real ),
                   cudaMemcpyHostToDevice ) != cudaSuccess )
   {
      checkCUDAError( __FILE__, __LINE__ );
      cerr << "Transfer of data from CUDA host ( " << rhs_vector. getName()
           << " ) to CUDA device ( " << this -> getName() << " ) failed." << endl;
      cerr << "Data: " << this -> data << endl;
   }
   if( safeMode )
      cudaThreadSynchronize();
   return *this;
#else
   cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
   return *this;
#endif
};

template< typename Real, typename Index >
tnlLongVector< Real, tnlCuda, Index >& tnlLongVector< Real, tnlCuda, Index > :: operator = ( const tnlLongVector< Real, tnlCuda, Index >& rhs_vector )
{
#ifdef HAVE_CUDA
    tnlAssert( rhs_vector. getSize() == this -> getSize(),
            cerr << "Source name: " << rhs_vector. getName() << endl
                 << "Source size: " << rhs_vector. getSize() << endl
                 << "Target name: " << this -> getName() << endl
                 << "Target size: " << this -> getSize() << endl );

   if( cudaMemcpy( this -> data,
                   rhs_vector. getVector(),
                   this -> getSize() * sizeof( Real ),
                   cudaMemcpyDeviceToDevice ) != cudaSuccess )
   {
      checkCUDAError( __FILE__, __LINE__ );
      cerr << "Transfer of data on the CUDA device from ( " << rhs_vector. getName()
           << " ) to ( " << this -> getName() << " ) failed." << endl;
      return *this;
   }
   if( safeMode )
      cudaThreadSynchronize();
   return *this;
#else
   cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
   return *this;
#endif
};

template< typename Real, typename Index >
bool tnlLongVector< Real, tnlCuda, Index > :: save( tnlFile& file ) const
{
   tnlAssert( this -> size != 0,
              cerr << "You try to save empty vector. Its name is " << this -> getName() );
   if( ! tnlObject :: save( file ) )
      return false;
   if( ! file. write( &this -> size, 1 ) )
      return false;
   if( ! file. write( this -> data, this -> size, tnlCuda ) )
   {
      cerr << "I was not able to WRITE the long vector " << this -> getName()
           << " with size " << this -> getSize() << endl;
      return false;
   }
   //cerr << "Writing " << this -> size << " elements from " << this -> getName() << "." << endl;
   return true;
};

template< typename Real, typename Index >
bool tnlLongVector< Real, tnlCuda, Index > :: load( tnlFile& file )
{
   if( ! tnlObject :: load( file ) )
      return false;
   int _size;
   if( ! file. read( &_size, 1 ) )
      return false;
   if( _size <= 0 )
   {
      cerr << "Error: The size " << _size << " of the file is not a positive number." << endl;
      return false;
   }
   setSize( _size );
   if( ! file. read( this -> data, this -> size, tnlCuda ) )
   {
      cerr << "I was not able to READ the long vector " << this -> getName()
           << " with size " << this -> getSize() << endl;
      return false;
   }
   return true;
};

#ifdef HAVE_CUDA
template< typename Real, typename Index, typename Real2, typename Index2 >
__global__ void tnlLongVectorCUDAAssignOperatorKernel( const Real2* source,
                                                       const Index targetSize,
                                                       Real* target )
{
   const Index i = blockIdx. x * blockDim. x + threadIdx. x;
   if( i < targetSize )
      target[ i ] = ( Real ) source[ ( Index2 ) i ];
}
#endif

template< typename Real, typename Index >
   template< typename Real2, typename Index2 >
tnlLongVector< Real, tnlCuda, Index >& tnlLongVector< Real, tnlCuda, Index > :: operator = ( const tnlLongVector< Real2, tnlCuda, Index2 >& long_vector )
{
   tnlAssert( long_vector. getSize() == this -> getSize(),
           cerr << "Source name: " << long_vector. getName() << endl
                << "Source size: " << long_vector. getSize() << endl
                << "Target name: " << this -> getName() << endl
                << "Target size: " << this -> getSize() << endl );

   #ifdef HAVE_CUDA
   dim3 blockSize, gridSize;
   blockSize. x = 512;
   gridSize. x = this -> getSize() / 512 + 1;

   tnlLongVectorCUDAAssignOperatorKernel<<< gridSize, blockSize >>>( long_vector. getVector(),
                                                                     this -> getSize(),
                                                                     this -> getVector() );
#else
   cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
#endif
   return ( *this );
};

template< typename Real, typename Index >
void tnlLongVector< Real, tnlCuda, Index > :: setValue( const Real& v )
{
#ifdef HAVE_CUDA
   dim3 blockSize, gridSize;
   blockSize. x = 512;
   gridSize. x = this -> getSize() / 512 + 1;

   tnlLongVectorCUDASetValueKernel<<< gridSize, blockSize >>>( this -> getVector(),
                                                               this -> getSize(),
                                                               v );
#else
   cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
#endif
}

template< typename Real, typename Index >
tnlLongVector< Real, tnlCuda, Index > :: ~tnlLongVector()
{
   dbgFunctionName( "tnlLongVectorCUDA", "~tnlLongVectorCUDA" );
#ifdef HAVE_CUDA
   if( this -> data && ! this -> shared_data )
   {
      dbgCout( "Freeing allocated memory of " << this -> getName() << " on CUDA device." );
      if( cudaFree( this -> data ) != cudaSuccess )
      {
         cerr << "Unable to free allocated memory on CUDA device of " << this -> getName() << "." << endl;
         checkCUDAError( __FILE__, __LINE__ );
      }
   }
#else
   //cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
#endif
};


#ifdef HAVE_CUDA
/****
 * TODO: !!! If the type Real would be some larger type this might fail.
 * CUDA cannot pass more than 256 bytes. We may need to pass the
 * parameter v explicitly.
 */
template< typename Real, typename Index >
__global__ void tnlLongVectorCUDASetValueKernel( Real* data,
                                                 const Index size,
                                                 const Real v )
{
   const Index i = blockIdx. x * blockDim. x + threadIdx. x;
   if( i < size )
      data[ i ] = v;
}
#endif

template< typename Real, typename Index >
bool operator == ( const tnlLongVector< Real, tnlHost, Index >& host_vector,
		             const tnlLongVector< Real, tnlCuda, Index >& cuda_vector )
{
	return ( cuda_vector == host_vector );
};

template< typename Real, typename Index >
bool operator != ( const tnlLongVector< Real, tnlHost, Index >& host_vector,
		             const tnlLongVector< Real, tnlCuda, Index >& cuda_vector )
{
	return !( cuda_vector == host_vector );
};

template< typename Real, typename Index >
ostream& operator << ( ostream& str, const tnlLongVector< Real, tnlCuda, Index >& vec )
{
#ifdef HAVE_CUDA
   Index size = vec. getSize();
   Real buffer[ 1024 ];
   Index pos( 0 );
   while( pos < size )
   {
	   int transfer = Min( size - pos, 1024 );
	   if( cudaMemcpy( buffer,
			             &( vec. getVector()[ pos ] ),
   		    		    transfer * sizeof( Real ), cudaMemcpyDeviceToHost ) != cudaSuccess )
	   {
		   cerr << "Transfer of data from CUDA device ( " << vec. getName()
   		        << " ) failed." << endl;
		   checkCUDAError( __FILE__, __LINE__ );
		   return str;
	   }
	   for( Index i = 0; i < transfer; i ++ )
		   str << buffer[ i ] <<  " ";
	   pos += transfer;
   }
#else
   cerr << "CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
#endif
   return str;

};

template< typename Real, typename Index >
Real tnlMax( const tnlLongVector< Real, tnlCuda, Index >& v )
{
   tnlAssert( v. getSize() != 0,
              cerr << "Vector name is " << v. getName() );
   Real result( 0 );
   tnlCUDALongVectorReduction< Real, Real, Index, tnlParallelReductionMax >( v. getSize(),
                                                                             v. getVector(),
                                                                             ( Real* ) NULL,
                                                                             result,
                                                                             ( Real ) 0 );
   return result;
}

template< typename Real, typename Index >
Real tnlMin( const tnlLongVector< Real, tnlCuda, Index >& v )
{
   tnlAssert( v. getSize() != 0,
              cerr << "Vector name is " << v. getName() );
   Real result( 0 );
   tnlCUDALongVectorReduction< Real, Real, Index, tnlParallelReductionMin >( v. getSize(),
                                                                             v. getVector(),
                                                                             ( Real* ) NULL,
                                                                             result,
                                                                             ( Real ) 0 );
   return result;
}

template< typename Real, typename Index >
Real tnlAbsMax( const tnlLongVector< Real, tnlCuda, Index >& v )
{
   tnlAssert( v. getSize() != 0,
              cerr << "Vector name is " << v. getName() );
   Real result( 0 );
   tnlCUDALongVectorReduction< Real, Real, Index, tnlParallelReductionAbsMax >( v. getSize(),
                                                                                v. getVector(),
                                                                                ( Real* ) NULL,
                                                                                result,
                                                                                ( Real ) 0 );
   return result;
}

template< typename Real, typename Index >
Real tnlAbsMin( const tnlLongVector< Real, tnlCuda, Index >& v )
{
   tnlAssert( v. getSize() != 0,
              cerr << "Vector name is " << v. getName() );
   Real result( 0 );
   tnlCUDALongVectorReduction< Real, Real, Index, tnlParallelReductionAbsMin >( v. getSize(),
                                                                                v. getVector(),
                                                                                ( Real* ) NULL,
                                                                                result,
                                                                                ( Real ) 0 );
   return result;
}

template< typename Real, typename Index >
Real tnlLpNorm( const tnlLongVector< Real, tnlCuda, Index >& v, const Real& p )
{
   tnlAssert( v. getSize() != 0,
              cerr << "Vector name is " << v. getName() );
   Real result( 0 );
   tnlCUDALongVectorReduction< Real, Real, Index, tnlParallelReductionLpNorm >( v. getSize(),
                                                                                v. getVector(),
                                                                                ( Real* ) NULL,
                                                                                result,
                                                                                p );
   return result;
}

template< typename Real, typename Index >
Real tnlSum( const tnlLongVector< Real, tnlCuda, Index >& v )
{
   tnlAssert( v. getSize() != 0,
              cerr << "Vector name is " << v. getName() );
   Real result( 0 );
   tnlCUDALongVectorReduction< Real, Real, Index, tnlParallelReductionSum >( v. getSize(),
                                                                             v. getVector(),
                                                                             ( Real* ) NULL,
                                                                             result,
                                                                             ( Real ) 0 );
   return result;
}

#ifdef HAVE_CUDA
template< typename Real, typename Index >
__global__ void tnlLongVectorCUDAScalaMultiplicationKernel( const Index size,
                                                            const Real alpha,
                                                            Real* x )
{
   Index tid = blockDim. x * blockIdx. x + threadIdx. x;
   if( tid < size )
      x[ tid ] *= alpha;
}
#endif

template< typename Real, typename Index >
Real tnlDifferenceMax( const tnlLongVector< Real, tnlCuda, Index >& u,
                       const tnlLongVector< Real, tnlCuda, Index >& v )
{
   tnlAssert( u. getSize() != 0,
              cerr << "Vector name is " << u. getName() );
   tnlAssert( u. getSize() == v. getSize(),
              cerr << "Vector names are " << u. getName() << " and " << v. getName() );

   tnlAssert( false, ); // TODO: fix this
   /*Real result = u[ 0 ] - v[ 0 ];
   const Index n = v. getSize();
   for( Index i = 1; i < n; i ++ )
      result = Max( result, u[ i ] - v[ i ] );
   return result;*/
}

template< typename Real, typename Index >
Real tnlDifferenceMin( const tnlLongVector< Real, tnlCuda, Index >& u,
                       const tnlLongVector< Real, tnlCuda, Index >& v )
{
   tnlAssert( u. getSize() != 0,
              cerr << "Vector name is " << u. getName() );
   tnlAssert( u. getSize() == v. getSize(),
              cerr << "Vector names are " << u. getName() << " and " << v. getName() );

   tnlAssert( false, ); // TODO: fix this
   /*Real result = u[ 0 ] - v[ 0 ];
   const Index n = v. getSize();
   for( Index i = 1; i < n; i ++ )
      result = Min( result, u[ i ] - v[ i ] );
   return result;*/
}

template< typename Real, typename Index >
Real tnlDifferenceAbsMax( const tnlLongVector< Real, tnlCuda, Index >& u,
                          const tnlLongVector< Real, tnlCuda, Index >& v )
{
   tnlAssert( u. getSize() != 0,
              cerr << "Vector name is " << u. getName() );
   tnlAssert( u. getSize() == v. getSize(),
              cerr << "Vector names are " << u. getName() << " and " << v. getName() );

   tnlAssert( false, ); // TODO: fix this
   /*Real result = u[ 0 ] - v[ 0 ];
   const Index n = v. getSize();
   for( Index i = 1; i < n; i ++ )
      result = Max( result, ( Real ) fabs( u[ i ] - v[ i ] ) );
   return result;*/
}

template< typename Real, typename Index >
Real tnlDifferenceAbsMin( const tnlLongVector< Real, tnlCuda, Index >& u,
                          const tnlLongVector< Real, tnlCuda, Index >& v )
{
   tnlAssert( u. getSize() != 0,
              cerr << "Vector name is " << u. getName() );
   tnlAssert( u. getSize() == v. getSize(),
              cerr << "Vector names are " << u. getName() << " and " << v. getName() );

   tnlAssert( false, ); // TODO: fix this
   /*Real result = u[ 0 ] - v[ 0 ];
   const Index n = v. getSize();
   for( Index i = 1; i < n; i ++ )
      result = Min( result, ( Real ) fabs(  u[ i ] - v[ i ] ) );
   return result;*/
}

template< typename Real, typename Index >
Real tnlDifferenceLpNorm( const tnlLongVector< Real, tnlCuda, Index >& u,
                          const tnlLongVector< Real, tnlCuda, Index >& v, const Real& p )
{
   tnlAssert( u. getSize() != 0,
              cerr << "Vector name is " << u. getName() );
   tnlAssert( u. getSize() == v. getSize(),
              cerr << "Vector names are " << u. getName() << " and " << v. getName() );

   tnlAssert( false, ); // TODO: fix this
   /*const Index n = v. getSize();
   Real result = pow( ( Real ) fabs( u[ 0 ] - v[ 0 ] ), ( Real ) p );
   for( Index i = 1; i < n; i ++ )
      result += pow( ( Real ) fabs( u[ i ] - v[ i ] ), ( Real ) p  );
   return pow( result, 1.0 / p );*/
}

template< typename Real, typename Index >
Real tnlDifferenceSum( const tnlLongVector< Real, tnlCuda, Index >& u,
                       const tnlLongVector< Real, tnlCuda, Index >& v )
{
   tnlAssert( u. getSize() != 0,
              cerr << "Vector name is " << u. getName() );
   tnlAssert( u. getSize() == v. getSize(),
              cerr << "Vector names are " << u. getName() << " and " << v. getName() );

   tnlAssert( false, ); // TODO: fix this
   /*Real result = u[ 0 ] - v[ 0 ];
   const Index n = u. getSize();
   for( Index i = 1; i < n; i ++ )
      result += u[ i ] - v[ i ];
   return result;*/
};


template< typename Real, typename Index >
void tnlScalarMultiplication( const Real& alpha,
                              tnlLongVector< Real, tnlCuda, Index >& u )
{
   tnlAssert( u. getSize() != 0,
              cerr << "Vector name is " << u. getName() );
#ifdef HAVE_CUDA
   dim3 blockSize, gridSize;
   blockSize. x = 512;
   gridSize. x = u. getSize() / 512 + 1;

   tnlLongVectorCUDAScalaMultiplicationKernel<<< gridSize, blockSize >>>( u. getSize(),
                                                                          alpha,
                                                                          u. getVector() );
#else
   cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
#endif
}

template< typename Real, typename Index >
Real tnlSDOT( const tnlLongVector< Real, tnlCuda, Index >& u ,
              const tnlLongVector< Real, tnlCuda, Index >& v )
{
   tnlAssert( u. getSize() != 0,
              cerr << "Vector name is " << u. getName() );
   tnlAssert( u. getSize() == v. getSize(),
              cerr << "You try to compute SDOT of two vectors with different sizes." << endl
                   << "The first one is " << u. getName() << " with size " << u. getSize() << endl
                   << "The second one is " << v. getName() << " with size " << v. getSize() << endl );
   Real result( 0 );
   tnlCUDALongVectorReduction< Real, Real, Index, tnlParallelReductionSdot >( u. getSize(),
                                                                              u. getVector(),
                                                                              v. getVector(),
                                                                              result,
                                                                              ( Real ) 0 );
   return result;
}

#ifdef HAVE_CUDA
template< typename Real, typename Index >
__global__ void tnlLongVectorCUDASaxpyKernel( const Index size,
                                              const Real alpha,
                                              const Real* x,
                                              Real* y )
{
   Index tid = blockDim. x * blockIdx. x + threadIdx. x;
   if( tid < size )
      y[ tid ] += alpha * x[ tid ];
}
#endif

//! Compute SAXPY operation (Scalar Alpha X Plus Y ).
template< typename Real, typename Index >
void tnlSAXPY( const Real& alpha,
               const tnlLongVector< Real, tnlCuda, Index >& x,
               tnlLongVector< Real, tnlCuda, Index >& y )
{
   tnlAssert( x. getSize() != 0,
              cerr << "Vector name is " << x. getName() );
   tnlAssert( x. getSize() == y. getSize(),
              cerr << "You try to compute SAXPY for two vectors with different sizes." << endl
                   << "The first one is " << x. getName() << " with size " << x. getSize() << endl
                   << "The second one is " << y. getName() << " with size " << y. getSize() << endl );
#ifdef HAVE_CUDA
   dim3 blockSize, gridSize;
   blockSize. x = 512;
   gridSize. x = x. getSize() / 512 + 1;

   tnlLongVectorCUDASaxpyKernel<<< gridSize, blockSize >>>( x. getSize(),
                                                            alpha,
                                                            x. getVector(),
                                                            y. getVector() );
#else
   cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
#endif

}

#ifdef HAVE_CUDA
template< typename Real, typename Index >
__global__ void tnlLongVectorCUDASaxmyKernel( const Index size,
                                              const Real alpha,
                                              const Real* x,
                                              Real* y )
{
   Index tid = blockDim. x * blockIdx. x + threadIdx. x;
   if( tid < size )
      y[ tid ] = alpha * x[ tid ] - y[ tid ];
}
#endif

template< typename Real, typename Index >
void tnlSAXMY( const Real& alpha,
               const tnlLongVector< Real, tnlCuda, Index >& x,
               tnlLongVector< Real, tnlCuda, Index >& y )
{
   tnlAssert( x. getSize() != 0,
              cerr << "Vector name is " << x. getName() );
   tnlAssert( x. getSize() == y. getSize(),
              cerr << "You try to compute SAXPY for two vectors with different sizes." << endl
                   << "The first one is " << x. getName() << " with size " << x. getSize() << endl
                   << "The second one is " << y. getName() << " with size " << y. getSize() << endl );
#ifdef HAVE_CUDA
   dim3 blockSize, gridSize;
   blockSize. x = 512;
   gridSize. x = x. getSize() / 512 + 1;

   tnlLongVectorCUDASaxmyKernel<<< gridSize, blockSize >>>( x. getSize(),
                                                            alpha,
                                                            x. getVector(),
                                                            y. getVector() );
#else
   cerr << "I am sorry but CUDA support is missing on this system " << __FILE__ << " line " << __LINE__ << "." << endl;
#endif

}



#endif /* TNLLONGVECTORCUDA_H_ */
