/***************************************************************************
                          ReductionOperations.h  -  description
                             -------------------
    begin                : Mar 22, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <limits>  // std::numeric_limits

#include <TNL/Math.h>
#include <TNL/Devices/CudaCallable.h>

namespace TNL {
namespace Containers {
namespace Algorithms {

/*
 * Unary operations: reduction on one input vector.
 */

template< typename Data, typename Result = Data >
class ParallelReductionSum
{
public:
   using DataType1 = Data;
   using DataType2 = void;
   using ResultType = Result;
   using LaterReductionOperation = ParallelReductionSum< Result >;

   static constexpr Result initialValue() { return 0; };

   template< typename Index >
   __cuda_callable__ void
   firstReduction( ResultType& result,
                   const Index& index,
                   const DataType1* data1,
                   const DataType2* data2 )
   {
      result += data1[ index ];
   }

   __cuda_callable__ void
   commonReduction( ResultType& result,
                    const ResultType& data )
   {
      result += data;
   }

   __cuda_callable__ void
   commonReduction( volatile ResultType& result,
                    volatile const ResultType& data )
   {
      result += data;
   }
};

template< typename Data, typename Result = Data >
class ParallelReductionMin
{
public:
   using DataType1 = Data;
   using DataType2 = void;
   using ResultType = Result;
   using LaterReductionOperation = ParallelReductionMin< Result >;

   static constexpr Result initialValue() { return std::numeric_limits< Result >::max(); };

   template< typename Index >
   __cuda_callable__ void
   firstReduction( ResultType& result,
                   const Index& index,
                   const DataType1* data1,
                   const DataType2* data2 )
   {
      result = TNL::min( result, data1[ index ] );
   }

   __cuda_callable__ void
   commonReduction( ResultType& result,
                    const Result& data )
   {
      result = TNL::min( result, data );
   }

   __cuda_callable__ void
   commonReduction( volatile ResultType& result,
                    volatile const Result& data )
   {
      result = TNL::min( result, data );
   }
};

template< typename Data, typename Result = Data >
class ParallelReductionMax
{
public:
   using DataType1 = Data;
   using DataType2 = void;
   using ResultType = Result;
   using LaterReductionOperation = ParallelReductionMax< Result >;

   static constexpr Result initialValue() { return std::numeric_limits< Result >::lowest(); };

   template< typename Index >
   __cuda_callable__ void
   firstReduction( ResultType& result,
                   const Index& index,
                   const DataType1* data1,
                   const DataType2* data2 )
   {
      result = TNL::max( result, data1[ index ] );
   }

   __cuda_callable__ void
   commonReduction( ResultType& result,
                    const Result& data )
   {
      result = TNL::max( result, data );
   }

   __cuda_callable__ void
   commonReduction( volatile ResultType& result,
                    volatile const Result& data )
   {
      result = TNL::max( result, data );
   }
};

template< typename Data, typename Result = bool >
class ParallelReductionLogicalAnd
{
public:
   using DataType1 = Data;
   using DataType2 = void;
   using ResultType = Result;
   using LaterReductionOperation = ParallelReductionLogicalAnd< Result >;

   static constexpr Result initialValue() { return true; };

   template< typename Index >
   __cuda_callable__ void
   firstReduction( ResultType& result,
                   const Index& index,
                   const DataType1* data1,
                   const DataType2* data2 )
   {
      result = result && data1[ index ];
   }

   __cuda_callable__ void
   commonReduction( ResultType& result,
                    const Result& data )
   {
      result = result && data;
   }

   __cuda_callable__ void
   commonReduction( volatile ResultType& result,
                    volatile const Result& data )
   {
      result = result && data;
   }
};


template< typename Data, typename Result = bool >
class ParallelReductionLogicalOr
{
public:
   using DataType1 = Data;
   using DataType2 = void;
   using ResultType = Result;
   using LaterReductionOperation = ParallelReductionLogicalOr< Result >;

   static constexpr Result initialValue() { return false; };

   template< typename Index >
   __cuda_callable__ void
   firstReduction( ResultType& result,
                   const Index& index,
                   const DataType1* data1,
                   const DataType2* data2 )
   {
      result = result || data1[ index ];
   }

   __cuda_callable__ void
   commonReduction( ResultType& result,
                    const Result& data )
   {
      result = result || data;
   }

   __cuda_callable__ void
   commonReduction( volatile ResultType& result,
                    volatile const Result& data )
   {
      result = result || data;
   }
};

template< typename Data, typename Result = Data >
class ParallelReductionAbsSum : public ParallelReductionSum< Data, Result >
{
public:
   using DataType1 = Data;
   using DataType2 = void;
   using ResultType = Result;
   using LaterReductionOperation = ParallelReductionSum< Result >;

   static constexpr Result initialValue() { return 0; };

   template< typename Index >
   __cuda_callable__ void
   firstReduction( ResultType& result,
                   const Index& index,
                   const DataType1* data1,
                   const DataType2* data2 )
   {
      result += TNL::abs( data1[ index ] );
   }
};

template< typename Data, typename Result = Data >
class ParallelReductionAbsMin : public ParallelReductionMin< Data, Result >
{
public:
   using DataType1 = Data;
   using DataType2 = void;
   using ResultType = Result;
   using LaterReductionOperation = ParallelReductionMin< Result >;

   static constexpr Result initialValue() { return std::numeric_limits< Result >::max(); };

   template< typename Index >
   __cuda_callable__ void
   firstReduction( ResultType& result,
                   const Index& index,
                   const DataType1* data1,
                   const DataType2* data2 )
   {
      result = TNL::min( result, TNL::abs( data1[ index ] ) );
   }
};

template< typename Data, typename Result = Data >
class ParallelReductionAbsMax : public ParallelReductionMax< Data, Result >
{
public:
   using DataType1 = Data;
   using DataType2 = void;
   using ResultType = Result;
   using LaterReductionOperation = ParallelReductionMax< Result >;

   static constexpr Result initialValue() { return 0; };

   template< typename Index >
   __cuda_callable__ void
   firstReduction( ResultType& result,
                   const Index& index,
                   const DataType1* data1,
                   const DataType2* data2 )
   {
      result = TNL::max( result, TNL::abs( data1[ index ] ) );
   }
};

template< typename Data, typename Result = Data >
class ParallelReductionL2Norm : public ParallelReductionSum< Data, Result >
{
public:
   using DataType1 = Data;
   using DataType2 = void;
   using ResultType = Result;
   using LaterReductionOperation = ParallelReductionSum< Result >;

   static constexpr Result initialValue() { return 0; };

   template< typename Index >
   __cuda_callable__ void
   firstReduction( ResultType& result,
                   const Index& index,
                   const DataType1* data1,
                   const DataType2* data2 )
   {
      const Data& aux = data1[ index ];
      result += aux * aux;
   }
};


template< typename Data, typename Result = Data, typename PType = Data >
class ParallelReductionLpNorm : public ParallelReductionSum< Data, Result >
{
public:
   using DataType1 = Data;
   using DataType2 = void;
   using ResultType = Result;
   using LaterReductionOperation = ParallelReductionSum< Result >;

   void setPower( const PType p )
   {
      this->p = p;
   }

   static constexpr Result initialValue() { return 0; };

   template< typename Index >
   __cuda_callable__ void
   firstReduction( ResultType& result,
                   const Index& index,
                   const DataType1* data1,
                   const DataType2* data2 )
   {
      result += TNL::pow( TNL::abs( data1[ index ] ), p );
   }

protected:
   PType p;
};


/*
 * Binary operations: reduction on two input vectors.
 */

template< typename Data1, typename Data2, typename Result = bool >
class ParallelReductionEqualities : public ParallelReductionLogicalAnd< Result >
{
public:
   using DataType1 = Data1;
   using DataType2 = Data2;
   using ResultType = Result;
   using LaterReductionOperation = ParallelReductionLogicalAnd< Result >;

   static constexpr Result initialValue() { return true; };

   template< typename Index >
   __cuda_callable__ void
   firstReduction( ResultType& result,
                   const Index& index,
                   const DataType1* data1,
                   const DataType2* data2 )
   {
      result = result && ( data1[ index ] == data2[ index ] );
   }
};

template< typename Data1, typename Data2, typename Result = bool >
class ParallelReductionInequalities : public ParallelReductionLogicalAnd< Result >
{
public:
   using DataType1 = Data1;
   using DataType2 = Data2;
   using ResultType = Result;
   using LaterReductionOperation = ParallelReductionLogicalAnd< Result >;

   static constexpr Result initialValue() { return false; };

   template< typename Index >
   __cuda_callable__ void
   firstReduction( ResultType& result,
                   const Index& index,
                   const DataType1* data1,
                   const DataType2* data2 )
   {
      result = result && ( data1[ index ] != data2[ index ] );
   }
};

template< typename Data1, typename Data2, typename Result = Data1 >
class ParallelReductionScalarProduct : public ParallelReductionSum< Result, Result >
{
public:
   using DataType1 = Data1;
   using DataType2 = Data2;
   using ResultType = Result;
   using LaterReductionOperation = ParallelReductionSum< Result >;

   static constexpr Result initialValue() { return 0; };

   template< typename Index >
   __cuda_callable__ void
   firstReduction( ResultType& result,
                   const Index& index,
                   const DataType1* data1,
                   const DataType2* data2 )
   {
      result += data1[ index ] * data2[ index ];
   }
};

template< typename Data1, typename Data2, typename Result = Data1 >
class ParallelReductionDiffSum : public ParallelReductionSum< Result, Result >
{
public:
   using DataType1 = Data1;
   using DataType2 = Data2;
   using ResultType = Result;
   using LaterReductionOperation = ParallelReductionSum< Result >;

   static constexpr Result initialValue() { return 0; };

   template< typename Index >
   __cuda_callable__ void
   firstReduction( ResultType& result,
                   const Index& index,
                   const DataType1* data1,
                   const DataType2* data2 )
   {
      result += data1[ index ] - data2[ index ];
   }
};

template< typename Data1, typename Data2, typename Result = Data1 >
class ParallelReductionDiffMin : public ParallelReductionMin< Result, Result >
{
public:
   using DataType1 = Data1;
   using DataType2 = Data2;
   using ResultType = Result;
   using LaterReductionOperation = ParallelReductionMin< Result >;

   static constexpr Result initialValue() { return std::numeric_limits< Result >::max(); };

   template< typename Index >
   __cuda_callable__ void
   firstReduction( ResultType& result,
                   const Index& index,
                   const DataType1* data1,
                   const DataType2* data2 )
   {
      result = TNL::min( result, data1[ index ] - data2[ index ] );
   }
};

template< typename Data1, typename Data2, typename Result = Data1 >
class ParallelReductionDiffMax : public ParallelReductionMax< Result, Result >
{
public:
   using DataType1 = Data1;
   using DataType2 = Data2;
   using ResultType = Result;
   using LaterReductionOperation = ParallelReductionMax< Result >;

   static constexpr Result initialValue() { return std::numeric_limits< Result >::lowest(); };

   template< typename Index >
   __cuda_callable__ void
   firstReduction( ResultType& result,
                   const Index& index,
                   const DataType1* data1,
                   const DataType2* data2 )
   {
      result = TNL::max( result, data1[ index ] - data2[ index ] );
   }
};

template< typename Data1, typename Data2, typename Result = Data1 >
class ParallelReductionDiffAbsSum : public ParallelReductionMax< Result, Result >
{
public:
   using DataType1 = Data1;
   using DataType2 = Data2;
   using ResultType = Result;
   using LaterReductionOperation = ParallelReductionSum< Result >;

   static constexpr Result initialValue() { return 0; };

   template< typename Index >
   __cuda_callable__ void
   firstReduction( ResultType& result,
                   const Index& index,
                   const DataType1* data1,
                   const DataType2* data2 )
   {
      result += TNL::abs( data1[ index ] - data2[ index ] );
   }
};

template< typename Data1, typename Data2, typename Result = Data1 >
class ParallelReductionDiffAbsMin : public ParallelReductionMin< Result, Result >
{
public:
   using DataType1 = Data1;
   using DataType2 = Data2;
   using ResultType = Result;
   using LaterReductionOperation = ParallelReductionMin< Result >;

   static constexpr Result initialValue() { return std::numeric_limits< Result >::max(); };

   template< typename Index >
   __cuda_callable__ void
   firstReduction( ResultType& result,
                   const Index& index,
                   const DataType1* data1,
                   const DataType2* data2 )
   {
      result = TNL::min( result, TNL::abs( data1[ index ] - data2[ index ] ) );
   }
};

template< typename Data1, typename Data2, typename Result = Data1 >
class ParallelReductionDiffAbsMax : public ParallelReductionMax< Result, Result >
{
public:
   using DataType1 = Data1;
   using DataType2 = Data2;
   using ResultType = Result;
   using LaterReductionOperation = ParallelReductionMax< Result >;

   static constexpr Result initialValue() { return 0; };

   template< typename Index >
   __cuda_callable__ void
   firstReduction( ResultType& result,
                   const Index& index,
                   const DataType1* data1,
                   const DataType2* data2 )
   {
      result = TNL::max( result, TNL::abs( data1[ index ] - data2[ index ] ) );
   }
};

template< typename Data1, typename Data2, typename Result = Data1 >
class ParallelReductionDiffL2Norm : public ParallelReductionSum< Result, Result >
{
public:
   using DataType1 = Data1;
   using DataType2 = Data2;
   using ResultType = Result;
   using LaterReductionOperation = ParallelReductionSum< Result >;

   static constexpr Result initialValue() { return 0; };

   template< typename Index >
   __cuda_callable__ void
   firstReduction( ResultType& result,
                   const Index& index,
                   const DataType1* data1,
                   const DataType2* data2 )
   {
      const ResultType aux = data2[ index ] - data1[ index ];
      result += aux * aux;
   }
};

template< typename Data1, typename Data2, typename Result = Data1, typename PType = Data1 >
class ParallelReductionDiffLpNorm : public ParallelReductionSum< Result, Result >
{
public:
   using DataType1 = Data1;
   using DataType2 = Data2;
   using ResultType = Result;
   using LaterReductionOperation = ParallelReductionSum< Result >;

   void setPower( const PType p )
   {
      this->p = p;
   }

   static constexpr Result initialValue() { return 0; };

   template< typename Index >
   __cuda_callable__ void
   firstReduction( ResultType& result,
                   const Index& index,
                   const DataType1* data1,
                   const DataType2* data2 )
   {
      result += TNL::pow( TNL::abs( data1[ index ] - data2[ index ] ), p );
   }

protected:
   PType p;
};

template< typename Data, typename Result = bool >
class ParallelReductionContainsValue : public ParallelReductionLogicalOr< Result >
{
public:
   using DataType1 = Data;
   using DataType2 = Data;
   using ResultType = Result;
   using LaterReductionOperation = ParallelReductionLogicalOr< Result >;

   template< typename Index >
   __cuda_callable__ void
   firstReduction( ResultType& result,
                   const Index& index,
                   const DataType1* data1,
                   const DataType2* data2 )
   {
      result = result || ( data1[ index ] == value );
   }

   void setValue( const Data& v )
   {
      this->value = v;
   }

protected:
   Data value;
};

template< typename Data, typename Result = bool >
class ParallelReductionContainsOnlyValue : public ParallelReductionLogicalAnd< Result >
{
public:
   using DataType1 = Data;
   using DataType2 = Data;
   using ResultType = Result;
   using LaterReductionOperation = ParallelReductionLogicalAnd< Result >;

   template< typename Index >
   __cuda_callable__ void
   firstReduction( ResultType& result,
                   const Index& index,
                   const DataType1* data1,
                   const DataType2* data2 )
   {
      result = result && ( data1[ index ] == value );
   }

   void setValue( const Data& v )
   {
      this->value = v;
   }

protected:
   Data value;
};

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
