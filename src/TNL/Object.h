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
 * \brief Basic class for majority of TNL objects like matrices, meshes, grids, solvers, etc..
 *
 * Objects like numerical meshes, matrices large vectors etc. are inherited by 
 * this class. This class introduces virtual method \ref getType which is 
 * supposed to tell the object type in a C++ style.
 */
class Object
{
   public:

      /**
       * \brief Static type getter.
       *
       * Returns the type in C++ style - for example the returned value
       * may look as \c "Array< double, Devices::Cuda, int >".
       * 
       * \par Example
       * \include ObjectExample_getType.cpp
       * \par Output
       * \include ObjectExample_getType.out
       */
      static String getType();

      /***
       * \brief Virtual type getter.
       * 
       * Returns the type in C++ style - for example the returned value
       * may look as \c "Array< double, Devices::Cuda, int >".
       * See example at \ref Object::getType.
       */
      virtual String getTypeVirtual() const;

      /**
       * \brief Static serialization type getter.
       *
       * Objects in TNL are saved as in a device independent manner. This method
       * is supposed to return the object type but with the device type replaced 
       * by Devices::Host. For example \c Array< double, Devices::Cuda > is
       * saved as \c Array< double, Devices::Host >.
       * See example at \ref Object::getType.
       */
      static String getSerializationType();

      /***
       * \brief Virtual serialization type getter.
       * 
       * Objects in TNL are saved as in a device independent manner. This method
       * is supposed to return the object type but with the device type replaced 
       * by Devices::Host. For example \c Array< double, Devices::Cuda > is
       * saved as \c Array< double, Devices::Host >.
       * See example at \ref Object::getType.
       */
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
#ifndef HAVE_MIC
      virtual ~Object(){};
#endif
};

bool getObjectType( File& file, String& type );

bool getObjectType( const String& file_name, String& type );

std::vector< String >
parseObjectType( const String& objectType );

} // namespace TNL

#include <TNL/Object.hpp>
