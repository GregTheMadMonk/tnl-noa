/***************************************************************************
                          tnlArrayIO.h  -  description
                             -------------------
    begin                : Mar 13, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/core/tnlDynamicTypeTag.h>
#include <TNL/core/tnlFile.h>

namespace TNL {

template< typename Element,
          typename Device,
          typename Index,
          bool DynamicType = tnlDynamicTypeTag< Element >::value >
class tnlArrayIO
{};

template< typename Element,
          typename Device,
          typename Index >
class tnlArrayIO< Element, Device, Index, true >
{
   public:

   static bool save( tnlFile& file,
                     const Element* data,
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

   static bool load( tnlFile& file,
                     Element* data,
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

template< typename Element,
          typename Device,
          typename Index >
class tnlArrayIO< Element, Device, Index, false >
{
   public:

   static bool save( tnlFile& file,
                     const Element* data,
                     const Index elements )
   {
      return file.write< Element, Device, Index >( data, elements );
   }

   static bool load( tnlFile& file,
                     Element* data,
                     const Index elements )
   {
      return file.read< Element, Device, Index >( data, elements );
   }

};

} // namespace TNL
