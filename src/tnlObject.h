/***************************************************************************
                          tnlObject.h  -  description
                             -------------------
    begin                : 2005/10/15
    copyright            : (C) 2005 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <core/tnlCuda.h>
#include <core/tnlString.h>


namespace TNL {

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
   public:

   //! Basic constructor
   __cuda_callable__
   tnlObject() {};

   /****
    * Type getter. This returns the type in C++ style - for example the returned value
    * may look as follows: "tnlVector< double, tnlCuda >".
    */
   static tnlString getType();

   virtual tnlString getTypeVirtual() const;

   /****
    * This is used for load and save methods.
    * Each object is saved as if it was stored on tnlHost. So even tnlVector< double, tnlCuda >
    * is saved as tnlVector< double, tnlHost >.
    */
   static tnlString getSerializationType();

   virtual tnlString getSerializationTypeVirtual() const;

   //! Method for saving the object to a file as a binary data
   virtual bool save( tnlFile& file ) const;

   //! Method for restoring the object from a file
   virtual bool load( tnlFile& file );
 
   //! Method for restoring the object from a file
   virtual bool boundLoad( tnlFile& file );

   bool save( const tnlString& fileName ) const;

   bool load( const tnlString& fileName );
 
   bool boundLoad( const tnlString& fileName );

   //! Destructor
   // FIXME: __cuda_callable__ would have to be added to every overriding destructor,
   // even if the object's constructor is not __cuda_callable__
//   __cuda_callable__
   virtual ~tnlObject(){};

};

bool getObjectType( tnlFile& file, tnlString& type );

bool getObjectType( const tnlString& file_name, tnlString& type );

bool parseObjectType( const tnlString& objectType,
                      tnlList< tnlString >& parsedObjectType );

} // namespace TNL