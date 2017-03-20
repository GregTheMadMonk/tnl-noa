/***************************************************************************
                          reduction-operations.h  -  description
                             -------------------
    begin                : Mar 22, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once 

#include <TNL/Constants.h>
#include <TNL/Math.h>
#include <TNL/Devices/Cuda.h>



namespace TNL {
namespace Containers {
namespace Algorithms {

template< typename Real, typename Index >
class tnlParallelReductionSum
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef Real ResultType;
   typedef tnlParallelReductionSum< Real, Index > LaterReductionOperation;

   ResultType reduceOnHost( const IndexType& idx,
                            const ResultType& current,
                            const RealType* data1,
                            const RealType* data2 )
   {
      return current + data1[ idx ];
   };
 
   __cuda_callable__ ResultType initialValue() { return 0; };
 
   __cuda_callable__ void cudaFirstReduction( ResultType& result,
                                              const IndexType& index,
                                              const RealType* data1,
                                              const RealType* data2 )
   {
      result += data1[ index ];
   }
 
   __cuda_callable__ void commonReductionOnDevice( ResultType& result,
                                                   const ResultType& data )
   {
      result += data;
   };
 
   __cuda_callable__ void commonReductionOnDevice( volatile ResultType& result,
                                                   volatile const ResultType& data )
   {
      result += data;
   };
};

template< typename Real, typename Index >
class tnlParallelReductionMin
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef Real ResultType;
   typedef tnlParallelReductionMin< Real, Index > LaterReductionOperation;

   ResultType reduceOnHost( const IndexType& idx,
                            const ResultType& current,
                            const RealType* data1,
                            const RealType* data2 )
   {
      return TNL::min( current, data1[ idx ] );
   };

   __cuda_callable__ ResultType initialValue() { return MaxValue< ResultType>(); };
 
   __cuda_callable__ void cudaFirstReduction( ResultType& result,
                                              const IndexType& index,
                                              const RealType* data1,
                                              const RealType* data2 )
   {
      result = TNL::min( result, data1[ index ] );
   }
 
   __cuda_callable__ void commonReductionOnDevice( ResultType& result,
                                                   const ResultType& data )
   {
      result = TNL::min( result, data );
   };
 
   __cuda_callable__ void commonReductionOnDevice( volatile ResultType& result,
                                                   volatile const ResultType& data )
   {
      result = TNL::min( result, data );
   };
};

template< typename Real, typename Index >
class tnlParallelReductionMax
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef Real ResultType;
   typedef tnlParallelReductionMax< Real, Index > LaterReductionOperation;

   ResultType reduceOnHost( const IndexType& idx,
                            const ResultType& current,
                            const RealType* data1,
                            const RealType* data2 )
   {
      return TNL::max( current, data1[ idx ] );
   };

   __cuda_callable__ ResultType initialValue() { return MinValue< ResultType>(); };
 
   __cuda_callable__ void cudaFirstReduction( ResultType& result,
                                              const IndexType& index,
                                              const RealType* data1,
                                              const RealType* data2 )
   {
      result = TNL::max( result, data1[ index ] );
   }
 
   __cuda_callable__ void commonReductionOnDevice( ResultType& result,
                                                   const ResultType& data )
   {
      result = TNL::max( result, data );
   };

   __cuda_callable__ void commonReductionOnDevice( volatile ResultType& result,
                                                   volatile const ResultType& data )
   {
      result = TNL::max( result, data );
   };
};

template< typename Real, typename Index >
class tnlParallelReductionLogicalAnd
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef Real ResultType;
   typedef tnlParallelReductionLogicalAnd< Real, Index > LaterReductionOperation;

   ResultType reduceOnHost( const IndexType& idx,
                            const ResultType& current,
                            const RealType* data1,
                            const RealType* data2 )
   {
      return current && data1[ idx ];
   };

   __cuda_callable__ ResultType initialValue() { return ( ResultType ) true; };
 
   __cuda_callable__ void cudaFirstReduction( ResultType& result,
                                              const IndexType& index,
                                              const RealType* data1,
                                              const RealType* data2 )
   {
      result = result && data1[ index ];
   }
 
   __cuda_callable__ void commonReductionOnDevice( ResultType& result,
                                                   const ResultType& data )
   {
      result = result && data;
   };
 
   __cuda_callable__ void commonReductionOnDevice( volatile ResultType& result,
                                                   volatile const ResultType& data )
   {
      result = result && data;
   };
};


template< typename Real, typename Index >
class tnlParallelReductionLogicalOr
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef Real ResultType;
   typedef tnlParallelReductionLogicalOr< Real, Index > LaterReductionOperation;

   ResultType reduceOnHost( const IndexType& idx,
                            const ResultType& current,
                            const RealType* data1,
                            const RealType* data2 )
   {
      return current || data1[ idx ];
   };
 
   __cuda_callable__ ResultType initialValue() { return ( ResultType ) false; };
 
   __cuda_callable__ void cudaFirstReduction( ResultType& result,
                                              const IndexType& index,
                                              const RealType* data1,
                                              const RealType* data2 )
   {
      result = result || data1[ index ];
   }

   __cuda_callable__ void commonReductionOnDevice( ResultType& result,
                                                   const ResultType& data )
   {
      result = result || data;
   };
 
   __cuda_callable__ void commonReductionOnDevice( volatile ResultType& result,
                                                   volatile const ResultType& data )
   {
      result = result || data;
   };
};

template< typename Real, typename Index >
class tnlParallelReductionAbsSum : public tnlParallelReductionSum< Real, Index >
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef Real ResultType;
   typedef tnlParallelReductionSum< Real, Index > LaterReductionOperation;

   ResultType reduceOnHost( const IndexType& idx,
                            const ResultType& current,
                            const RealType* data1,
                            const RealType* data2 )
   {
      return current + TNL::abs( data1[ idx ] );
   };

   __cuda_callable__ ResultType initialValue() { return ( ResultType ) 0; };

   __cuda_callable__ void cudaFirstReduction( ResultType& result,
                                              const IndexType& index,
                                              const RealType* data1,
                                              const RealType* data2 )
   {
      result += TNL::abs( data1[ index ] );
   }
};

template< typename Real, typename Index >
class tnlParallelReductionAbsMin : public tnlParallelReductionMin< Real, Index >
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef Real ResultType;
   typedef tnlParallelReductionMin< Real, Index > LaterReductionOperation;

   ResultType reduceOnHost( const IndexType& idx,
                            const ResultType& current,
                            const RealType* data1,
                            const RealType* data2 )
   {
      return TNL::min( current, TNL::abs( data1[ idx ] ) );
   };

   __cuda_callable__ ResultType initialValue() { return MaxValue< ResultType>(); };
 
   __cuda_callable__ void cudaFirstReduction( ResultType& result,
                                              const IndexType& index,
                                              const RealType* data1,
                                              const RealType* data2 )
   {
      result = TNL::min( result, TNL::abs( data1[ index ] ) );
   }
};

template< typename Real, typename Index >
class tnlParallelReductionAbsMax : public tnlParallelReductionMax< Real, Index >
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef Real ResultType;
   typedef tnlParallelReductionMax< Real, Index > LaterReductionOperation;

   ResultType reduceOnHost( const IndexType& idx,
                            const ResultType& current,
                            const RealType* data1,
                            const RealType* data2 )
   {
      return std::max( current, TNL::abs( data1[ idx ] ) );
   };

   __cuda_callable__ ResultType initialValue() { return ( ResultType ) 0; };

   __cuda_callable__ void cudaFirstReduction( ResultType& result,
                                              const IndexType& index,
                                              const RealType* data1,
                                              const RealType* data2 )
   {
      result = TNL::max( result, TNL::abs( data1[ index ] ) );
   }
};

template< typename Real, typename Index >
class tnlParallelReductionL2Norm : public tnlParallelReductionSum< Real, Index >
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef Real ResultType;
   typedef tnlParallelReductionSum< Real, Index > LaterReductionOperation;

   ResultType reduceOnHost( const IndexType& idx,
                            const ResultType& current,
                            const RealType* data1,
                            const RealType* data2 )
   {
      const RealType& aux = data1[ idx ];
      return current + aux * aux;
   };

   __cuda_callable__ ResultType initialValue() { return ( ResultType ) 0; };
 
   __cuda_callable__ void cudaFirstReduction( ResultType& result,
                                              const IndexType& index,
                                              const RealType* data1,
                                              const RealType* data2 )
   {
      const RealType& aux = data1[ index ];
      result += aux * aux;
   }
};


template< typename Real, typename Index >
class tnlParallelReductionLpNorm : public tnlParallelReductionSum< Real, Index >
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef Real ResultType;
   typedef tnlParallelReductionSum< Real, Index > LaterReductionOperation;

   void setPower( const RealType& p )
   {
      this->p = p;
   };

   ResultType reduceOnHost( const IndexType& idx,
                            const ResultType& current,
                            const RealType* data1,
                            const RealType* data2 )
   {
      return current + TNL::pow( TNL::abs( data1[ idx ] ), p );
   };

   __cuda_callable__ ResultType initialValue() { return ( ResultType ) 0; };
 
   __cuda_callable__ void cudaFirstReduction( ResultType& result,
                                              const IndexType& index,
                                              const RealType* data1,
                                              const RealType* data2 )
   {
      result += TNL::pow( TNL::abs( data1[ index ] ), p );
   }
 
   protected:

   RealType p;
};

template< typename Real, typename Index >
class tnlParallelReductionEqualities : public tnlParallelReductionLogicalAnd< bool, Index >
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef bool ResultType;
   typedef tnlParallelReductionLogicalAnd< bool, Index > LaterReductionOperation;

   ResultType reduceOnHost( const IndexType& idx,
                            const ResultType& current,
                            const RealType* data1,
                            const RealType* data2 )
   {
      return current && ( data1[ idx ] == data2[ idx ] );
   };

   __cuda_callable__ ResultType initialValue() { return ( ResultType ) true; };
 
   __cuda_callable__ void cudaFirstReduction( ResultType& result,
                                              const IndexType& index,
                                              const RealType* data1,
                                              const RealType* data2 )
   {
      result = result && ( data1[ index ] == data2[ index ] );
   }
};

template< typename Real, typename Index >
class tnlParallelReductionInequalities : public tnlParallelReductionLogicalAnd< bool, Index >
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef bool ResultType;
   typedef tnlParallelReductionLogicalAnd< bool, Index > LaterReductionOperation;

   ResultType reduceOnHost( const IndexType& idx,
                            const ResultType& current,
                            const RealType* data1,
                            const RealType* data2 )
   {
      return current && ( data1[ idx ] != data2[ idx ] );
   };

   __cuda_callable__ ResultType initialValue() { return ( ResultType ) false; };
 
   __cuda_callable__ void cudaFirstReduction( ResultType& result,
                                              const IndexType& index,
                                              const RealType* data1,
                                              const RealType* data2 )
   {
      result = result && ( data1[ index ] != data2[ index ] );
   }
};

template< typename Real, typename Index >
class tnlParallelReductionScalarProduct : public tnlParallelReductionSum< Real, Index >
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef Real ResultType;
   typedef tnlParallelReductionSum< Real, Index > LaterReductionOperation;

   ResultType reduceOnHost( const IndexType& idx,
                            const ResultType& current,
                            const RealType* data1,
                            const RealType* data2 )
   {
      return current + ( data1[ idx ] * data2[ idx ] );
   };

   __cuda_callable__ ResultType initialValue() { return ( ResultType ) 0; };
 
   __cuda_callable__ inline void cudaFirstReduction( ResultType& result,
                                                     const IndexType& index,
                                                     const RealType* data1,
                                                     const RealType* data2 )
   {
      result += data1[ index ] * data2[ index ];
   }
};

template< typename Real, typename Index >
class tnlParallelReductionDiffSum : public tnlParallelReductionSum< Real, Index >
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef Real ResultType;
   typedef tnlParallelReductionSum< Real, Index > LaterReductionOperation;

   ResultType reduceOnHost( const IndexType& idx,
                            const ResultType& current,
                            const RealType* data1,
                            const RealType* data2 )
   {
      return current + ( data1[ idx ] - data2[ idx ] );
   };
 
   __cuda_callable__ ResultType initialValue() { return ( ResultType ) 0; };
 
   __cuda_callable__ void cudaFirstReduction( ResultType& result,
                                              const IndexType& index,
                                              const RealType* data1,
                                              const RealType* data2 )
   {
      result += data1[ index ] - data2[ index ];
   }
};

template< typename Real, typename Index >
class tnlParallelReductionDiffMin : public tnlParallelReductionMin< Real, Index >
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef Real ResultType;
   typedef tnlParallelReductionMin< Real, Index > LaterReductionOperation;

   ResultType reduceOnHost( const IndexType& idx,
                            const ResultType& current,
                            const RealType* data1,
                            const RealType* data2 )
   {
      return TNL::min( current, data1[ idx ] - data2[ idx ] );
   };

   __cuda_callable__ ResultType initialValue() { return MaxValue< ResultType>(); };
 
   __cuda_callable__ void cudaFirstReduction( ResultType& result,
                                              const IndexType& index,
                                              const RealType* data1,
                                              const RealType* data2 )
   {
      result = TNL::min( result, data1[ index ] - data2[ index ] );
   }
};

template< typename Real, typename Index >
class tnlParallelReductionDiffMax : public tnlParallelReductionMax< Real, Index >
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef Real ResultType;
   typedef tnlParallelReductionMax< Real, Index > LaterReductionOperation;

   ResultType reduceOnHost( const IndexType& idx,
                            const ResultType& current,
                            const RealType* data1,
                            const RealType* data2 )
   {
      return TNL::max( current, data1[ idx ] - data2[ idx ] );
   };

   __cuda_callable__ ResultType initialValue() { return ( ResultType ) 0; };
 
   __cuda_callable__ void cudaFirstReduction( ResultType& result,
                                              const IndexType& index,
                                              const RealType* data1,
                                              const RealType* data2 )
   {
      result = TNL::max( result, data1[ index ] - data2[ index ] );
   }
};

template< typename Real, typename Index >
class tnlParallelReductionDiffAbsSum : public tnlParallelReductionMax< Real, Index >
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef Real ResultType;
   typedef tnlParallelReductionSum< Real, Index > LaterReductionOperation;

   ResultType reduceOnHost( const IndexType& idx,
                            const ResultType& current,
                            const RealType* data1,
                            const RealType* data2 )
   {
      return current + TNL::abs( data1[ idx ] - data2[ idx ] );
   };

   __cuda_callable__ ResultType initialValue() { return ( ResultType ) 0; };
 
   __cuda_callable__ void cudaFirstReduction( ResultType& result,
                                              const IndexType& index,
                                              const RealType* data1,
                                              const RealType* data2 )
   {
      result += TNL::abs( data1[ index ] - data2[ index ] );
   }
};

template< typename Real, typename Index >
class tnlParallelReductionDiffAbsMin : public tnlParallelReductionMin< Real, Index >
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef Real ResultType;
   typedef tnlParallelReductionMin< Real, Index > LaterReductionOperation;

   ResultType reduceOnHost( const IndexType& idx,
                            const ResultType& current,
                            const RealType* data1,
                            const RealType* data2 )
   {
      return TNL::min( current, TNL::abs( data1[ idx ] - data2[ idx ] ) );
   };

   __cuda_callable__ ResultType initialValue() { return MaxValue< ResultType>(); };
 
   __cuda_callable__ void cudaFirstReduction( ResultType& result,
                                              const IndexType& index,
                                              const RealType* data1,
                                              const RealType* data2 )
   {
      result = TNL::min( result, TNL::abs( data1[ index ] - data2[ index ] ) );
   }
};

template< typename Real, typename Index >
class tnlParallelReductionDiffAbsMax : public tnlParallelReductionMax< Real, Index >
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef Real ResultType;
   typedef tnlParallelReductionMax< Real, Index > LaterReductionOperation;

   ResultType reduceOnHost( const IndexType& idx,
                            const ResultType& current,
                            const RealType* data1,
                            const RealType* data2 )
   {
      return TNL::max( current, TNL::abs( data1[ idx ] - data2[ idx ] ) );
   };

   __cuda_callable__ ResultType initialValue() { return ( ResultType ) 0; };
 
   __cuda_callable__ void cudaFirstReduction( ResultType& result,
                                              const IndexType& index,
                                              const RealType* data1,
                                              const RealType* data2 )
   {
      result = TNL::max( result, TNL::abs( data1[ index ] - data2[ index ] ) );
   }
};

template< typename Real, typename Index >
class tnlParallelReductionDiffL2Norm : public tnlParallelReductionSum< Real, Index >
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef Real ResultType;
   typedef tnlParallelReductionSum< Real, Index > LaterReductionOperation;

   ResultType reduceOnHost( const IndexType& idx,
                            const ResultType& current,
                            const RealType* data1,
                            const RealType* data2 )
   {
      this->aux = data2[ idx ] - data1[ idx ];
      return current + aux * aux;
   };

   __cuda_callable__ ResultType initialValue() { return ( ResultType ) 0; };
 
   __cuda_callable__ void cudaFirstReduction( ResultType& result,
                                              const IndexType& index,
                                              const RealType* data1,
                                              const RealType* data2 )
   {
      this->aux = data2[ index ] - data1[ index ];
      result += aux * aux;
   }
 
   protected:
 
      RealType aux;
};

template< typename Real, typename Index >
class tnlParallelReductionDiffLpNorm : public tnlParallelReductionSum< Real, Index >
{
   public:

   typedef Real RealType;
   typedef Index IndexType;
   typedef Real ResultType;
   typedef tnlParallelReductionSum< Real, Index > LaterReductionOperation;

   void setPower( const RealType& p )
   {
      this->p = p;
   };

   ResultType reduceOnHost( const IndexType& idx,
                            const ResultType& current,
                            const RealType* data1,
                            const RealType* data2 )
   {
      return current + TNL::pow( TNL::abs( data1[ idx ] - data2[ idx ] ), p );
   };

   __cuda_callable__ ResultType initialValue() { return ( ResultType ) 0; };
 
   __cuda_callable__ void cudaFirstReduction( ResultType& result,
                                              const IndexType& index,
                                              const RealType* data1,
                                              const RealType* data2 )
   {
      result += TNL::pow( TNL::abs( data1[ index ] - data2[ index ] ), p );
   }
 
   protected:

   RealType p;
};

} // namespace Algorithms
} // namespace Containers
} // namespace TNL

