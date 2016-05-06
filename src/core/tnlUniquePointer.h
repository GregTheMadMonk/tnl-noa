/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

/***************************************************************************
                          tnlUniquePointer.h  -  description
                             -------------------
    begin                : May 6, 2016
    copyright            : (C) 2016 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

#pragma once

#include <core/tnlHost.h>
#include <core/tnlCuda.h>

template< typename Object, typename Device = typename Object::DeviceType >
class tnlUniquePointer
{  
};

template< typename Object, typename Device, typename... Args >
tnlUniquePointer< Object, Device >& makeUniquePointer( Args&& args )
{
   return
}

template< typename Object >
class tnlUniquePointer< Object, tnlHost >
{
   public:
      
      typedef Object ObjectType;
      typedef tnlHost DeviceType;
      typedef tnlUniquePointer< Object, tnlHost > ThisType;
         
      tnlUniquePointer()
      : pointer( 0 )
      {         
      }
      
      template< typename... Args >
      void create( const Args&& args )
      {
         this->pointer = new Object( args );
      }
      
      const Object& operator->() const
      {
         return *( this->pointer );
      }
      
      Object& operator->()
      {
         return *( this->pointer );
      }
      
      const Object& operator*() const
      {
         return *( this->pointer );
      }
      
      Object& operator*()
      {
         return *( this->pointer );
      }
      
      const Object& get() const
      {
         return *( this->pointer );
      }

      Object& modify()
      {
         return *( this->pointer );
      }
      
      const ThisType& operator=( ThisType& ptr )
      {
         if( this-> pointer )
            delete this->pointer;
      }
      
      ~tnlUniquePointer()
      {
         if( this->pointer )
            delete this->pointer;
      }

      
   protected:
      
      Object* pointer;
      
};

