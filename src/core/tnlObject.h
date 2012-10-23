/***************************************************************************
                          tnlObject.h  -  description
                             -------------------
    begin                : 2005/10/15
    copyright            : (C) 2005 by Tomas Oberhuber
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

#ifndef tnlObjectH
#define tnlObjectH

#include <core/tnlString.h>

class tnlFile;
template< class T > class tnlList;

//! This is basic class for all 'large' objects like matrices, meshes, grids, solvers etc.
/*!
 *  Objects like numerical grids, meshes, matrices large vectors etc.
 *  are inherited by this class. This class provides name for such objects. Giving
 *  a name to each bigger object is compulsory. The name can help to locate
 *  possible errors in the code. This can help to identify an object where, for
 *  example, one tries to touch non-existing element. All objects of the TNL should
 *  have only constructor with name and then only setter methods and method init.
 *  Each object derived from the tnlObject must be able to tell its type via the method getType and
 *  it must support methods for saving and loading the object from a file.
 */
class tnlObject
{
   private:
   //! Constructor with no parameters is not allowed.
   tnlObject();

   public:

   //! Constructor with name
   tnlObject( const tnlString& name );

   /****
    * Type getter. This returns the type in C++ style - for example the returned value
    * may look ass follows: "tnlVector< double, tnlCuda >".
    */
   virtual tnlString getType() const;

   /****
    *  Name getter
    */
   const tnlString& getName() const;

   //! Method for saving the object to a file as a binary data
   virtual bool save( tnlFile& file ) const;

   //! Method for restoring the object from a file
   virtual bool load( tnlFile& file );

   bool save( const tnlString& fileName ) const;

   bool load( const tnlString& fileName );

   //! Destructor
   virtual ~tnlObject(){};

   protected:

   //! Object name
   tnlString name;
};

bool getObjectType( tnlFile& file, tnlString& type );

bool getObjectType( const tnlString& file_name, tnlString& type );

bool parseObjectType( const tnlString& objectType,
                      tnlList< tnlString >& parsedObjectType );

#endif
