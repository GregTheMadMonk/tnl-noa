/***************************************************************************
                          MPITypeResolver.h  -  description
                             -------------------
    begin                : Feb 4, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

namespace TNL {
namespace Communicators {

#ifdef HAVE_MPI
template<typename Type>
struct MPITypeResolver
{
   static inline MPI_Datatype getType()
   {
      static_assert( sizeof(Type) == sizeof(char) ||
                     sizeof(Type) == sizeof(int) ||
                     sizeof(Type) == sizeof(short int) ||
                     sizeof(Type) == sizeof(long int),
                     "Fatal Error - Unknown MPI Type");
      switch( sizeof( Type ) )
      {
         case sizeof( char ):
            return MPI_CHAR;
         case sizeof( int ):
            return MPI_INT;
         case sizeof( short int ):
            return MPI_SHORT;
         case sizeof( long int ):
            return MPI_LONG;
      }
   };
};

template<> struct MPITypeResolver< char >
{
    static inline MPI_Datatype getType(){return MPI_CHAR;};
};

template<> struct MPITypeResolver< int >
{
    static inline MPI_Datatype getType(){return MPI_INT;};
};

template<> struct MPITypeResolver< short int >
{
    static inline MPI_Datatype getType(){return MPI_SHORT;};
};

template<> struct MPITypeResolver< long int >
{
    static inline MPI_Datatype getType(){return MPI_LONG;};
};

template<> struct MPITypeResolver< unsigned char >
{
    static inline MPI_Datatype getType(){return MPI_UNSIGNED_CHAR;};
};

template<> struct MPITypeResolver< unsigned short int >
{
    static inline MPI_Datatype getType(){return MPI_UNSIGNED_SHORT;};
};

template<> struct MPITypeResolver< unsigned int >
{
    static inline MPI_Datatype getType(){return MPI_UNSIGNED;};
};

template<> struct MPITypeResolver< unsigned long int >
{
    static inline MPI_Datatype getType(){return MPI_UNSIGNED_LONG;};
};

template<> struct MPITypeResolver< float >
{
    static inline MPI_Datatype getType(){return MPI_FLOAT;};
};

template<> struct MPITypeResolver< double >
{
    static inline MPI_Datatype getType(){return MPI_DOUBLE;};
};

template<> struct MPITypeResolver< long double >
{
    static inline MPI_Datatype getType(){return MPI_LONG_DOUBLE;};
};

template<> struct MPITypeResolver< bool >
{
   // sizeof(bool) is implementation-defined: https://stackoverflow.com/a/4897859
   static_assert( sizeof(bool) == 1, "The systems where sizeof(bool) != 1 are not supported by MPI." );
   static inline MPI_Datatype getType() { return MPI_C_BOOL; };
};
#endif

} // namespace Communicators
} // namespace TNL
