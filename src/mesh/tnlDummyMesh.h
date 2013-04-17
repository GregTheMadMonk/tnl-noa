/***************************************************************************
                          tnlDummyMesh.h  -  description
                             -------------------
    begin                : Apr 17, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
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

#ifndef TNLDUMMYMESH_H_
#define TNLDUMMYMESH_H_

template< typename Real, typename Device, typename Index >
class tnlDummyMesh
{
   public:

   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   bool save( tnlFile& file ) const{};

   //! Method for restoring the object from a file
   bool load( tnlFile& file ){};

   bool save( const tnlString& fileName ) const{};

   bool load( const tnlString& fileName ){};

   template< typename MeshFunction >
   bool write( const MeshFunction& function,
                const tnlString& fileName,
                const tnlString& format ) const{};
};


#endif /* TNLDUMMYMESH_H_ */
