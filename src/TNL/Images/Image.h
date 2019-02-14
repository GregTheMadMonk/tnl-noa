/***************************************************************************
                          Image.h  -  description
                             -------------------
    begin                : Jul 20, 2015
    copyright            : (C) 2015 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
/**
 * \brief Namespace for image processing.
 */
namespace Images {

template< typename Index = int >
class Image
{
   public:
 
      typedef Index IndexType;
 
      Image() : width( 0 ), height( 0 ) {};
 
      IndexType getWidth() const
      {
         return this->width;
      }
 
      IndexType getHeight() const
      {
         return this->height;
      }
 
   protected:
 
      IndexType width, height;
 
};

} // namespace Images
} // namespace TNL

