/***************************************************************************
                          tnlArrayIO.h  -  description
                             -------------------
    begin                : Mar 13, 2014
    copyright            : (C) 2014 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLARRAYIO_H_
#define TNLARRAYIO_H_

#include<core/tnlDynamicTypeTag.h>
#include<core/tnlFile.h>

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
            cerr << "I was not able to save " << i << "-th of " << elements << " elements." << endl;
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
            cerr << "I was not able to load " << i << "-th of " << elements << " elements." << endl;
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

#endif /* TNLARRAYIO_H_ */
