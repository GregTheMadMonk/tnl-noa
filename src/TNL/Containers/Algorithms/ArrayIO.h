/***************************************************************************
                          ArrayIO.h  -  description
                             -------------------
    begin                : Mar 13, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <type_traits>

#include <TNL/Object.h>
#include <TNL/File.h>

namespace TNL {
namespace Containers {
namespace Algorithms {

template< typename Value,
          typename Device,
          typename Index,
          bool Elementwise = std::is_base_of< Object, Value >::value >
class ArrayIO
{};

template< typename Value,
          typename Device,
          typename Index >
class ArrayIO< Value, Device, Index, true >
{
   public:

   static bool save( File& file,
                     const Value* data,
                     const Index elements )
   {
      for( Index i = 0; i < elements; i++ )
         if( ! data[ i ].save( file ) )
         {
            std::cerr << "I was not able to save " << i << "-th of " << elements << " elements." << std::endl;
            return false;
         }
      return true;
   }

   static bool load( File& file,
                     Value* data,
                     const Index elements )
   {
      for( Index i = 0; i < elements; i++ )
         if( ! data[ i ].load( file ) )
         {
            std::cerr << "I was not able to load " << i << "-th of " << elements << " elements." << std::endl;
            return false;
         }
      return true;
   }
};

template< typename Value,
          typename Device,
          typename Index >
class ArrayIO< Value, Device, Index, false >
{
   public:

   static bool save( File& file,
                     const Value* data,
                     const Index elements )
   {
      return file.save< Value, Value, Device >( data, elements );
   }

   static bool load( File& file,
                     Value* data,
                     const Index elements )
   {
      return file.load< Value, Value, Device >( data, elements );
   }

};

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
