/***************************************************************************
                          Object.h  -  description
                             -------------------
    begin                : 2005/10/15
    copyright            : (C) 2005 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Devices/Cuda.h>
#include <TNL/String.h>
#include <TNL/File.h>
#include <TNL/Containers/List.h>

namespace TNL {

//! This is basic class for all 'large' objects like matrices, meshes, grids, solvers etc.
/*!
 *  Objects like numerical grids, meshes, matrices large vectors etc.
 *  are inherited by this class. This class provides name for such objects. Giving
 *  a name to each bigger object is compulsory. The name can help to locate
 *  possible errors in the code. This can help to identify an object where, for
 *  example, one tries to touch non-existing element. All objects of the TNL should
 *  have only constructor with name and then only setter methods and method init.
 *  Each object derived from the Object must be able to tell its type via the method getType and
 *  it must support methods for saving and loading the object from a file.
 */
class Object
{
   public:

      //! Basic constructor
#ifndef HAVE_MIC
      __cuda_callable__
#endif
      Object() : deprecatedReadMode( false ) {};

      /****
       * Type getter. This returns the type in C++ style - for example the returned value
       * may look as follows: "Vector< double, Devices::Cuda >".
       */

      static String getType();      

      virtual String getTypeVirtual() const;   

      /****
       * This is used for load and save methods.
       * Each object is saved as if it was stored on Devices::Host. So even Vector< double, Devices::Cuda >
       * is saved as Vector< double, Devices::Host >.
       */
      static String getSerializationType();

      virtual String getSerializationTypeVirtual() const;

      //! Method for saving the object to a file as a binary data
      virtual bool save( File& file ) const;

      //! Method for restoring the object from a file
      virtual bool load( File& file );

      //! Method for restoring the object from a file
      virtual bool boundLoad( File& file );

      bool save( const String& fileName ) const;

      bool load( const String& fileName );

      bool boundLoad( const String& fileName );
      
      void setDeprecatedReadMode();

      //! Destructor
      // FIXME: __cuda_callable__ would have to be added to every overriding destructor,
      // even if the object's constructor is not __cuda_callable__
      //   __cuda_callable__
      virtual ~Object(){};
   
   
   protected:
      
      // This allows to read old TNL files, it will be removed.
      bool deprecatedReadMode;

};

bool getObjectType( File& file, String& type );

bool getObjectType( const String& file_name, String& type );

bool parseObjectType( const String& objectType,
                      Containers::List< String >& parsedObjectType );

} // namespace TNL
