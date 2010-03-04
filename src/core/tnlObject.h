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

#include <core/tnlAssert.h>
#include <core/tnlString.h>

//! This is basic class for many other more complex objects.
/*! Objects like numerical grids, meshes, matrices large vectors etc.
    are inhereted by this class. This class provides name for such objects
    and methods for saving object in to a file and for later restoring. This
    is important for saving state of the computation for the case of system
    or program halt. Name is useful for debuging and for referencing objects
    during the restoring. 
*/
class tnlObject
{
   public:

   //! Basic constructor
   tnlObject(); // TODO: Remove this we want to have only constructor with name

   //! Constructor with name
   tnlObject( const char* name );
   
   //! Copy constructor 
   /*! It does not copy name - name should be unique
    */
   tnlObject( const tnlObject& object );

   //! Type getter
   virtual tnlString GetType() const = 0;

   //! Name setter
   void SetName( const char* _name );

   //! Name getter
   const tnlString& GetName() const;

   //! Method for saving the object to a file as a binary data
   virtual bool Save( ostream& file ) const;   

   //! Method for restoring the object from a file
   virtual bool Load( istream& file );   

   //! Destructor
   virtual ~tnlObject(){};

   protected:

   //! Object name
   tnlString name;
};

bool GetObjectType( istream& file, tnlString& type );

bool GetObjectType( const char* file_name, tnlString& type );

#endif
