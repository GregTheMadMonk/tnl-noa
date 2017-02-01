/***************************************************************************
                          MeshSimplexTopology.h  -  description
                             -------------------
    begin                : Aug 29, 2015
    copyright            : (C) 2015 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Zabka Vitezslav, zabkav@gmail.com
 */


#pragma once

#include <TNL/String.h>

namespace TNL {
namespace Meshes {

template< int dimension_ >
class MeshSimplexTopology
{
   public:
      static constexpr int dimension = dimension_;

      static String getType()
      {
         return String( "MeshSimplexTopology< " ) + String( dimension ) + " >";
      }
};

template< unsigned int n, unsigned int k >
class tnlStaticNumCombinations;

template<unsigned int n, unsigned int k, unsigned int combinationIndex, unsigned int valueIndex>
class tnlCombinationValue;

template< int dimension,
          int subtopologyDim >
class MeshSubtopology< MeshSimplexTopology< dimension >, subtopologyDim >
{
   static_assert( 0 < subtopologyDim && subtopologyDim < dim, "invalid subtopology dimension" );

   static constexpr int topologyVertexCount = MeshSubtopology< MeshSimplexTopology< dimension >, 0 >::count;
   static constexpr int subtopologyVertexCount = MeshSubtopology< MeshSimplexTopology< subtopologyDim >, 0>::count;

   public:
      typedef MeshSimplexTopology< subtopologyDim > Topology;

      static constexpr int count = tnlNumCombinations< topologyVertexCount, subtopologyVertexCount >::value;
};

template< int dimension >
class MeshSubtopology< MeshSimplexTopology< dimension >, 0 >
{
   static_assert(0 < dim, "invalid dimension");

   public:
      typedef MeshVertexTopology Topology;

      static constexpr int count = dim + 1;
};


template< int dimension,
          typename Subtopology,
          int subtopologyIndex,
          int vertexIndex >
struct SubentityVertexMap< MeshSimplexTopology< dimension >, Subtopology, subtopologyIndex, vertexIndex >
{
   private:
      static constexpr int subtopologyCount = Subtopology< MeshSimplexTopology< dimension >, Subtopology::dimension >::count;
      static constexpr int topologyVertexCount = Subtopology< MeshSimplex< dimension >, 0 >::count;
      static constexpr int subtopologyVertexCount = Subtopology< Subtopology, 0 >::count;

      static_assert(1 < dimension, "subtopology vertex can be specified for topologies of dimension 2 or higher");
      static_assert(0 <= subtopologyIndex && subtopologyIndex < subtopologyCount, "invalid subtopology index");
      static_assert(0 <= vertexIndex && vertexIndex < subtopologyVertexCount, "invalid subtopology vertex index");

   public:
      static constexpr int index = CombinationValue< topologyVertexCount, subtopologyVertexCount, subtopologyIndex, vertexIndex>::value;
};

template< unsigned int n, unsigned int k >
class tnlStaticNumCombinations
{
   static_assert(0 < k && k < n, "invalid argument");

   public:
      static const unsigned int value = tnlNumCombinations< n - 1, k - 1 >::value + tnlNumCombinations< n - 1, k >::value;
};

// Nummber of combinations (n choose k)
template< unsigned int n >
class tnlNumCombinations< n, 0 >
{
   static_assert(0 <= n, "invalid argument");

   public:
      static const unsigned int value = 1;
};

template< unsigned int n >
class tnlNumCombinations< n, n >
{
   static_assert(0 < n, "invalid argument");

   public:
      static const unsigned int value = 1;
};

//     Compile-time generation of combinations
// Combinations are generated in lexicographical order. The following example shows generation of 2-combinations from set {0, 1, 2}:
//   0, 1  <->  CombinationValue<3, 2, 0, 0>::VALUE, CombinationValue<3, 2, 0, 1>::VALUE
//   0, 2  <->  CombinationValue<3, 2, 1, 0>::VALUE, CombinationValue<3, 2, 1, 1>::VALUE
//   1, 2  <->  CombinationValue<3, 2, 2, 0>::VALUE, CombinationValue<3, 2, 2, 1>::VALUE
template< unsigned int n,
          unsigned int k,
          unsigned int combinationIndex >
class tnlCombinationIncrement;

template< unsigned int n,
          unsigned int k,
          unsigned int combinationIndex,
          unsigned int valueIndex >
class tnlCombinationValue
{
   static_assert( combinationIndex < NumCombinations< n, k >::value, "invalid combination index" );
   static_assert( valueIndex < k, "invalid value index" );

   static const unsigned int incrementValueIndex = tnlCombinationIncrement< n, k, combinationIndex - 1>::valueIndex;

   public:
      static const unsigned int value = ( valueIndex < incrementValueIndex ? tnlCombinationValue< n, k, combinationIndex - 1, valueIndex >::value :
                                          tnlCombinationValue< n, k, combinationIndex - 1, incrementValueIndex >::value +
                                          valueIndex - incrementValueIndex + 1);
};

template< unsigned int n,
          unsigned int k,
          unsigned int valueIndex >
class tnlCombinationValue< n, k, 0, valueIndex >
{
   static_assert( valueIndex < k, "invalid value index" );

   static const unsigned int incrementValueIndex = tnlCombinationIncrement< n, k, 0 >::valueIndex;

   public:
      static const unsigned int value = valueIndex;
};

// The CombinationIncrement class determines value index of the particular combination which will be incremented when generating the next combination
template< unsigned int n,
          unsigned int k,
          unsigned int combinationIndex,
          unsigned int valueIndex >
class tnlCombinationIncrementImpl
{
   static_assert( combinationIndex < tnlNumCombinations< n, k >::value - 1, "nothing to increment" );

   static const bool incrementPossible = ( tnlCombinationValue< n, k, combinationIndex, valueIndex >::value + k - valueIndex < n );

   public:
      static constexpr int valueIndex = ( incrementPossible ? valueIndex : tnlCombinationIncrementImpl< n, k, combinationIndex, valueIndex - 1 >::valueIndex );
};

template< unsigned int n,
          unsigned int k,
          unsigned int combinationIndex >
class tnlCombinationIncrementImpl< n, k, combinationIndex, 0 >
{
   static_assert( combinationIndex < tnlNumCombinations<n, k>::value - 1, "nothing to increment" );

   public:
      static constexpr int valueIndex = 0;
};

template< unsigned int n,
          unsigned int k,
          unsigned int combinationIndex >
class tnlCombinationIncrement
{
   static_assert( combinationIndex < tnlNumCombinations< n, k >::value - 1, "nothing to increment" );

   public:
      static const unsigned int valueIndex = tnlCombinationIncrementImpl< n, k, combinationIndex, k - 1 >::valueIndex;
};

} // namespace Meshes
} // namespace TNL

