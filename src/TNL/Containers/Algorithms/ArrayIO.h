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
struct ArrayIO
{};

template< typename Value,
          typename Device,
          typename Index >
struct ArrayIO< Value, Device, Index, true >
{
   static String getSerializationType()
   {
      return String( "Containers::Array< " ) +
             TNL::getSerializationType< Value >() + ", " +
             TNL::getSerializationType< Devices::Host >() + ", " +
             TNL::getSerializationType< Index >() + " >";
   }

   static void save( File& file,
                     const Value* data,
                     const Index elements )
   {
      Index i;
      try
      {
         for( i = 0; i < elements; i++ )
            data[ i ].save( file );
      }
      catch(...)
      {
         throw Exceptions::FileSerializationError( file.getFileName(), "unable to write the " + std::to_string(i) + "-th array element of " + std::to_string(elements) + " into the file." );
      }
   }

   static void load( File& file,
                     Value* data,
                     const Index elements )
   {
      Index i = 0;
      try
      {
         for( i = 0; i < elements; i++ )
            data[ i ].load( file );
      }
      catch(...)
      {
         throw Exceptions::FileDeserializationError( file.getFileName(), "unable to read the " + std::to_string(i) + "-th array element of " + std::to_string(elements) + " from the file." );
      }
   }
};

template< typename Value,
          typename Device,
          typename Index >
struct ArrayIO< Value, Device, Index, false >
{
   static String getSerializationType()
   {
      return String( "Containers::Array< " ) +
             TNL::getSerializationType< Value >() + ", " +
             TNL::getSerializationType< Devices::Host >() + ", " +
             TNL::getSerializationType< Index >() + " >";
   }

   static void save( File& file,
                     const Value* data,
                     const Index elements )
   {
      if( elements == 0 )
         return;
      try
      {
         file.save< Value, Value, Device >( data, elements );
      }
      catch(...)
      {
         throw Exceptions::FileSerializationError( file.getFileName(), "unable to write " + std::to_string(elements) + " array elements into the file." );
      }
   }

   static void load( File& file,
                     Value* data,
                     const Index elements )
   {
      if( elements == 0 )
         return;
      try
      {
         file.load< Value, Value, Device >( data, elements );
      }
      catch(...)
      {
         throw Exceptions::FileDeserializationError( file.getFileName(), "unable to read " + std::to_string(elements) + " array elements from the file." );
      }
   }
};

} // namespace Algorithms
} // namespace Containers
} // namespace TNL
