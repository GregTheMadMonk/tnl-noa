/***************************************************************************
                          SimplexTopology.h  -  description
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

#include <TNL/Meshes/Topologies/MeshEntityTopology.h>
#include <TNL/Meshes/Topologies/MeshVertexTopology.h>

namespace TNL {
namespace Meshes {
namespace Topologies {

template< int dimension_ >
class Simplex
{
   public:
      static constexpr int dimension = dimension_;

      static String getType()
      {
         return String( "SimplexTopology< " ) + String( dimension ) + " >";
      }
};

namespace SimplexDetails {
   template< unsigned int n, unsigned int k >
   class NumCombinations;

   template<unsigned int n, unsigned int k, unsigned int combinationIndex, unsigned int valueIndex>
   class CombinationValue;
}

template< int dimension,
          int subtopologyDim >
class Subtopology< Simplex< dimension >, subtopologyDim >
{
   static_assert( 0 < subtopologyDim && subtopologyDim < dimension, "invalid subtopology dimension" );

   static constexpr int topologyVertexCount = Subtopology< Simplex< dimension >, 0 >::count;
   static constexpr int subtopologyVertexCount = Subtopology< Simplex< subtopologyDim >, 0>::count;

   public:
      typedef Simplex< subtopologyDim > Topology;

      static constexpr int count = SimplexDetails::NumCombinations< topologyVertexCount, subtopologyVertexCount >::value;
};

template< int dimension >
class Subtopology< Simplex< dimension >, 0 >
{
   static_assert(0 < dimension, "invalid dimension");

   public:
      typedef Vertex Topology;

      static constexpr int count = dimension + 1;
};


template< int dimension,
          typename Subtopology,
          int subtopologyIndex,
          int vertexIndex >
struct SubentityVertexMap< Simplex< dimension >, Subtopology, subtopologyIndex, vertexIndex >
{
   private:
      static constexpr int subtopologyCount = Topologies::Subtopology< Simplex< dimension >, Subtopology::dimension >::count;
      static constexpr int topologyVertexCount = Topologies::Subtopology< Simplex< dimension >, 0 >::count;
      static constexpr int subtopologyVertexCount = Topologies::Subtopology< Subtopology, 0 >::count;

      static_assert(1 < dimension, "subtopology vertex can be specified for topologies of dimension 2 or higher");
      static_assert(0 <= subtopologyIndex && subtopologyIndex < subtopologyCount, "invalid subtopology index");
      static_assert(0 <= vertexIndex && vertexIndex < subtopologyVertexCount, "invalid subtopology vertex index");

   public:
      static constexpr int index = SimplexDetails::CombinationValue< topologyVertexCount, subtopologyVertexCount, subtopologyIndex, vertexIndex>::value;
};

template< unsigned int n, unsigned int k >
class NumCombinations
{
   static_assert(0 < k && k < n, "invalid argument");

   public:
      static const unsigned int value = NumCombinations< n - 1, k - 1 >::value + NumCombinations< n - 1, k >::value;
};


namespace SimplexDetails {

// Nummber of combinations (n choose k)
template< unsigned int n >
class NumCombinations< n, 0 >
{
   static_assert(0 <= n, "invalid argument");

   public:
      static const unsigned int value = 1;
};

template< unsigned int n >
class NumCombinations< n, n >
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
class CombinationIncrement;

template< unsigned int n,
          unsigned int k,
          unsigned int combinationIndex,
          unsigned int valueIndex >
class CombinationValue
{
   static_assert( combinationIndex < NumCombinations< n, k >::value, "invalid combination index" );
   static_assert( valueIndex < k, "invalid value index" );

   static const unsigned int incrementValueIndex = CombinationIncrement< n, k, combinationIndex - 1>::valueIndex;

   public:
      static const unsigned int value = ( valueIndex < incrementValueIndex ? CombinationValue< n, k, combinationIndex - 1, valueIndex >::value :
                                          CombinationValue< n, k, combinationIndex - 1, incrementValueIndex >::value +
                                          valueIndex - incrementValueIndex + 1);
};

template< unsigned int n,
          unsigned int k,
          unsigned int valueIndex >
class CombinationValue< n, k, 0, valueIndex >
{
   static_assert( valueIndex < k, "invalid value index" );

   static const unsigned int incrementValueIndex = CombinationIncrement< n, k, 0 >::valueIndex;

   public:
      static const unsigned int value = valueIndex;
};

// The CombinationIncrement class determines value index of the particular combination which will be incremented when generating the next combination
template< unsigned int n,
          unsigned int k,
          unsigned int combinationIndex,
          unsigned int valueIndex_ >
class CombinationIncrementImpl
{
   static_assert( combinationIndex < NumCombinations< n, k >::value - 1, "nothing to increment" );

   static const bool incrementPossible = ( CombinationValue< n, k, combinationIndex, valueIndex_ >::value + k - valueIndex_ < n );

   public:
      static constexpr int valueIndex = ( incrementPossible ? valueIndex_ : CombinationIncrementImpl< n, k, combinationIndex, valueIndex_ - 1 >::valueIndex );
};

template< unsigned int n,
          unsigned int k,
          unsigned int combinationIndex >
class CombinationIncrementImpl< n, k, combinationIndex, 0 >
{
   static_assert( combinationIndex < NumCombinations<n, k>::value - 1, "nothing to increment" );

   public:
      static constexpr int valueIndex = 0;
};

template< unsigned int n,
          unsigned int k,
          unsigned int combinationIndex >
class CombinationIncrement
{
   static_assert( combinationIndex < NumCombinations< n, k >::value - 1, "nothing to increment" );

   public:
      static const unsigned int valueIndex = CombinationIncrementImpl< n, k, combinationIndex, k - 1 >::valueIndex;
};

} // namespace SimplexDetails

} // namespace Topologies
} // namespace Meshes
} // namespace TNL
