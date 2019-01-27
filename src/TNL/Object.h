/***************************************************************************
                          Object.h  -  description
                             -------------------
    begin                : 2005/10/15
    copyright            : (C) 2005 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <vector>

#include <TNL/Devices/CudaCallable.h>
#include <TNL/String.h>
#include <TNL/File.h>

/**
 * \brief The main TNL namespace.
 */
namespace TNL {

/**
 * \brief This is the basic class for all 'large' objects like matrices, meshes, grids, solvers, etc..
 *
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

      /**
       * \brief Type getter.
       *
       * Returns the type in C++ style - for example the returned value
       * may look as follows: "Vector< double, Devices::Cuda >".
       */
      static String getType();      

      virtual String getTypeVirtual() const;   

      /**
       * \brief This is used for load and save methods.
       *
       * Each object is saved as if it was stored on Devices::Host. So even Vector< double, Devices::Cuda >
       * is saved as Vector< double, Devices::Host >.
       */
      static String getSerializationType();

      virtual String getSerializationTypeVirtual() const;

      /**
       * \brief Method for saving the object to a file as a binary data.
       *
       * \param file Name of file object.
       */
      virtual bool save( File& file ) const;

      /**
       * \brief Method for restoring the object from a file.
       *
       * \param file Name of file object.
       */
      virtual bool load( File& file );

      /**
       * \brief Method for restoring the object from a file.
       *
       * \param file Name of file object.
       */
      virtual bool boundLoad( File& file );

      /**
       * \brief Method for saving the object to a file as a binary data.
       *
       * \param fileName String defining the name of a file.
       */
      bool save( const String& fileName ) const;

      /**
       * \brief Method for restoring the object from a file.
       *
       * \param fileName String defining the name of a file.
       */
      bool load( const String& fileName );

       /**
       * \brief Method for restoring the object from a file.
       *
       * \param fileName String defining the name of a file.
       */
      bool boundLoad( const String& fileName );
      
      /// Destructor.
      // FIXME: __cuda_callable__ would have to be added to every overriding destructor,
      // even if the object's constructor is not __cuda_callable__
      //   __cuda_callable__
#ifndef HAVE_MIC
      virtual ~Object(){};
#endif
};

bool getObjectType( File& file, String& type );

bool getObjectType( const String& file_name, String& type );

std::vector< String >
parseObjectType( const String& objectType );

} // namespace TNL

#include <TNL/Object_impl.h>
