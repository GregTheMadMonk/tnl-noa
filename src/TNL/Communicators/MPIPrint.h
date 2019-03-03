/***************************************************************************
                          MPIPrint.h  -  description
                             -------------------
    begin                : Feb 7, 2019
    copyright            : (C) 2019 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <sstream>
#include <TNL/Communicators/MpiCommunicator.h>

#ifdef HAVE_MPI
#define TNL_MPI_PRINT( message )                                                                                                 \
if( ! TNL::Communicators::MpiCommunicator::IsInitialized() )                                                                     \
   std::cerr << message << std::endl;                                                                                            \
else                                                                                                                             \
{                                                                                                                                \
   if( TNL::Communicators::MpiCommunicator::GetRank() > 0 )                                                                      \
   {                                                                                                                             \
      std::stringstream __tnl_mpi_print_stream_;                                                                                 \
      __tnl_mpi_print_stream_ << "Node " << TNL::Communicators::MpiCommunicator::GetRank() << " of "                             \
         << TNL::Communicators::MpiCommunicator::GetSize() << " : " << message << std::endl;                                     \
      TNL::String __tnl_mpi_print_string_( __tnl_mpi_print_stream_.str().c_str() );                                              \
      send( __tnl_mpi_print_string_, 0, std::numeric_limits< int >::max() );                                                                                         \
   }                                                                                                                             \
   else                                                                                                                          \
   {                                                                                                                             \
      std::cerr << "Node 0 of " << TNL::Communicators::MpiCommunicator::GetSize() << " : " << message << std::endl;              \
      for( int __tnl_mpi_print_j = 1;                                                                                            \
           __tnl_mpi_print_j < TNL::Communicators::MpiCommunicator::GetSize();                                                   \
           __tnl_mpi_print_j++ )                                                                                                 \
         {                                                                                                                       \
            TNL::String __tnl_mpi_print_string_;                                                                                 \
            receive( __tnl_mpi_print_string_, __tnl_mpi_print_j, std::numeric_limits< int >::max() );                                                                \
            std::cerr << __tnl_mpi_print_string_;                                                                                \
         }                                                                                                                       \
   }                                                                                                                             \
}
#else
#define TNL_MPI_PRINT( message )                                                                                                 \
   std::cerr << message << std::endl;
#endif

#ifdef HAVE_MPI
#define TNL_MPI_PRINT_MASTER( message )                                                                                          \
if( ! TNL::Communicators::MpiCommunicator::IsInitialized() )                                                                     \
   std::cerr << message << std::endl;                                                                                            \
else                                                                                                                             \
{                                                                                                                                \
   if( TNL::Communicators::MpiCommunicator::GetRank() == 0 )                                                                     \
   {                                                                                                                             \
      std::cerr << "Master node : " << message << std::endl;                                                                     \
   }                                                                                                                             \
}
#else
#define TNL_MPI_PRINT_MASTER( message )                                                                                          \
   std::cerr << message << std::endl;
#endif

#ifdef HAVE_MPI
#define TNL_MPI_PRINT_COND( condition, message )                                                                                 \
if( ! TNL::Communicators::MpiCommunicator::IsInitialized() )                                                                     \
{                                                                                                                                \
   if( condition) std::cerr << message << std::endl;                                                                             \
}                                                                                                                                \
else                                                                                                                             \
{                                                                                                                                \
   if( TNL::Communicators::MpiCommunicator::GetRank() > 0 )                                                                      \
   {                                                                                                                             \
      int __tnl_mpi_print_cnd = ( condition );                                                                                   \
      TNL::Communicators::MpiCommunicator::Send( &__tnl_mpi_print_cnd, 1, 0, 0 );                                                \
      if( condition ) {                                                                                                          \
         std::stringstream __tnl_mpi_print_stream_;                                                                              \
         __tnl_mpi_print_stream_ << "Node " << TNL::Communicators::MpiCommunicator::GetRank() << " of "                          \
            << TNL::Communicators::MpiCommunicator::GetSize() << " : " << message << std::endl;                                  \
         TNL::String __tnl_mpi_print_string_( __tnl_mpi_print_stream_.str().c_str() );                                           \
         send( __tnl_mpi_print_string_, 0, std::numeric_limits< int >::max() );                                                                                      \
      }                                                                                                                          \
   }                                                                                                                             \
   else                                                                                                                          \
   {                                                                                                                             \
      if( condition )                                                                                                            \
         std::cerr << "Node 0 of " << TNL::Communicators::MpiCommunicator::GetSize() << " : " << message << std::endl;           \
      for( int __tnl_mpi_print_j = 1;                                                                                            \
           __tnl_mpi_print_j < TNL::Communicators::MpiCommunicator::GetSize();                                                   \
           __tnl_mpi_print_j++ )                                                                                                 \
         {                                                                                                                       \
            int __tnl_mpi_print_cond;                                                                                            \
            TNL::Communicators::MpiCommunicator::Recv( &__tnl_mpi_print_cond, 1, __tnl_mpi_print_j, 0 );                         \
            if( __tnl_mpi_print_cond )                                                                                           \
            {                                                                                                                    \
               TNL::String __tnl_mpi_print_string_;                                                                              \
               receive( __tnl_mpi_print_string_, __tnl_mpi_print_j, std::numeric_limits< int >::max() );                                                             \
               std::cerr << __tnl_mpi_print_string_;                                                                             \
            }                                                                                                                    \
         }                                                                                                                       \
   }                                                                                                                             \
}
#else
#define TNL_MPI_PRINT_COND( condition, message )                                                                                 \
   std::cerr << message << std::endl;
#endif