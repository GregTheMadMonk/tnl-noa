/***************************************************************************
                          SizesHolder.h  -  description
                             -------------------
    begin                : Dec 24, 2018
    copyright            : (C) 2018 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

// Implemented by: Jakub Klinkovsky

#pragma once

#include <TNL/Assert.h>
#include <TNL/Devices/CudaCallable.h>
#include <TNL/TemplateStaticFor.h>

#include <TNL/Containers/ndarray/Meta.h>

namespace TNL {
namespace Containers {

namespace __ndarray_impl {

template< typename Index,
          typename LevelTag,
          std::size_t size >
class SizeHolder
{
public:
   __cuda_callable__
   constexpr Index getSize( LevelTag ) const
   {
      return size;
   }

   void setSize( LevelTag, Index newSize )
   {
      TNL_ASSERT( newSize == 0, );
   }

   __cuda_callable__
   bool operator==( const SizeHolder& ) const
   {
      return true;
   }
};

template< typename Index,
          typename LevelTag >
class SizeHolder< Index, LevelTag, 0 >
{
public:
   __cuda_callable__
   Index getSize( LevelTag ) const
   {
      return size;
   }

   void setSize( LevelTag, Index size )
   {
      TNL_ASSERT( size >= 0, );
      this->size = size;
   }

   __cuda_callable__
   bool operator==( const SizeHolder& other ) const
   {
      return size == other.size;
   }

private:
   Index size = 0;
};

template< typename Index,
          std::size_t currentSize,
          std::size_t... otherSizes >
class SizesHolderLayer
: public SizesHolderLayer< Index, otherSizes... >,
  public SizeHolder< Index,
                     IndexTag< sizeof...( otherSizes ) >,  // LevelTag
                     currentSize >
{
   using BaseType = SizesHolderLayer< Index, otherSizes... >;
   using Layer = SizeHolder< Index,
                             IndexTag< sizeof...( otherSizes ) >,  // LevelTag
                             currentSize >;
protected:
   using BaseType::getSize;
   using BaseType::setSize;
   using Layer::getSize;
   using Layer::setSize;

   __cuda_callable__
   bool operator==( const SizesHolderLayer& other ) const
   {
      return BaseType::operator==( other ) &&
             Layer::operator==( other );
   }
};

// specializations to terminate the recursive inheritance
template< typename Index,
          std::size_t currentSize >
class SizesHolderLayer< Index, currentSize >
: public SizeHolder< Index,
                     IndexTag< 0 >,  // LevelTag
                     currentSize >
{
    using Layer = SizeHolder< Index,
                              IndexTag< 0 >,  // LevelTag
                              currentSize >;
protected:
    using Layer::getSize;
    using Layer::setSize;

    __cuda_callable__
    bool operator==( const SizesHolderLayer& other ) const
    {
        return Layer::operator==( other );
    }
};

template< std::size_t dimension >
struct SizesHolderStaticSizePrinter
{
   template< typename SizesHolder >
   static void exec( std::ostream& str, const SizesHolder& holder )
   {
      str << holder.template getStaticSize< dimension >() << ", ";
   }
};

template< std::size_t dimension >
struct SizesHolderSizePrinter
{
   template< typename SizesHolder >
   static void exec( std::ostream& str, const SizesHolder& holder )
   {
      str << holder.template getSize< dimension >() << ", ";
   }
};

} // namespace __ndarray_impl


// dimensions and static sizes are specified as std::size_t,
// the type of dynamic sizes is configurable with Index

template< typename Index,
          std::size_t... sizes >
class SizesHolder
: public __ndarray_impl::SizesHolderLayer< Index, sizes... >
{
   using BaseType = __ndarray_impl::SizesHolderLayer< Index, sizes... >;

public:
   using IndexType = Index;

   static constexpr std::size_t getDimension()
   {
      return sizeof...( sizes );
   }

   template< std::size_t dimension >
   static constexpr std::size_t getStaticSize()
   {
      static_assert( dimension < sizeof...(sizes), "Invalid dimension passed to getStaticSize()." );
      return __ndarray_impl::get_from_pack< dimension >( sizes... );
   }

   template< std::size_t level >
   __cuda_callable__
   Index getSize() const
   {
      static_assert( level < sizeof...(sizes), "Invalid level passed to getSize()." );
      return BaseType::getSize( __ndarray_impl::IndexTag< getDimension() - level - 1 >() );
   }

   template< std::size_t level >
   void setSize( Index size )
   {
      static_assert( level < sizeof...(sizes), "Invalid level passed to setSize()." );
      BaseType::setSize( __ndarray_impl::IndexTag< getDimension() - level - 1 >(), size );
   }

   // methods for convenience
   __cuda_callable__
   bool operator==( const SizesHolder& other ) const
   {
      return BaseType::operator==( other );
   }

   __cuda_callable__
   bool operator!=( const SizesHolder& other ) const
   {
      return ! operator==( other );
   }
};


template< typename Index,
          std::size_t dimension,
          Index constSize >
class ConstStaticSizesHolder
{
public:
   using IndexType = Index;

   static constexpr std::size_t getDimension()
   {
      return dimension;
   }

   template< std::size_t level >
   static constexpr std::size_t getStaticSize()
   {
      static_assert( level < getDimension(), "Invalid level passed to getStaticSize()." );
      return constSize;
   }

   template< std::size_t level >
   __cuda_callable__
   Index getSize() const
   {
      static_assert( level < getDimension(), "Invalid dimension passed to getSize()." );
      return constSize;
   }

   // methods for convenience
   __cuda_callable__
   bool operator==( const ConstStaticSizesHolder& other ) const
   {
      return true;
   }

   __cuda_callable__
   bool operator!=( const ConstStaticSizesHolder& other ) const
   {
      return false;
   }
};


template< typename Index,
          std::size_t... sizes >
std::ostream& operator<<( std::ostream& str, const SizesHolder< Index, sizes... >& holder )
{
   str << "SizesHolder< ";
   TemplateStaticFor< std::size_t, 0, sizeof...(sizes) - 1, __ndarray_impl::SizesHolderStaticSizePrinter >::execHost( str, holder );
   str << holder.template getStaticSize< sizeof...(sizes) - 1 >() << " >( ";
   TemplateStaticFor< std::size_t, 0, sizeof...(sizes) - 1, __ndarray_impl::SizesHolderSizePrinter >::execHost( str, holder );
   str << holder.template getSize< sizeof...(sizes) - 1 >() << " )";
   return str;
}


// helper for the forInternal method
namespace __ndarray_impl {

template< typename SizesHolder,
          std::size_t ConstValue >
struct SubtractedSizesHolder
{};

template< typename Index,
          std::size_t ConstValue,
          std::size_t... sizes >
struct SubtractedSizesHolder< SizesHolder< Index, sizes... >, ConstValue >
{
//   using type = SizesHolder< Index, std::max( (std::size_t) 0, sizes - ConstValue )... >;
   using type = SizesHolder< Index, ( (sizes >= ConstValue) ? sizes - ConstValue : 0 )... >;
};

} // namespace __ndarray_impl

} // namespace Containers
} // namespace TNL
