/***************************************************************************
                          tnlMPIMesh2D.h  -  description
                             -------------------
    begin                : 2005/07/09
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

#ifndef tnlMPIMesh2DH
#define tnlMPIMesh2DH

#include <legacy/mesh/tnlGridOld.h>
#include <config/tnlParameterContainer.h>
#include <core/mpi-supp.h>
#include <debug/tnlDebug.h>
#include <core/mfilename.h>


template< typename Real, typename Device, typename Index >
class tnlMPIMesh< 2, Real, Device, Index >
{
   public:

   //! Basic constructor
   tnlMPIMesh();

   //! Destructor
   ~tnlMPIMesh()
   {
      FreeBuffers();
   };

   //! Initiation
   bool Init( const tnlGridOld< 2, Real, Device, Index >& u,
              int& _mesh_x_size,
              int& _mesh_y_size,
              Index _overlap_width,
              int root = 0,
              MPI_Comm comm = MPI_COMM_WORLD );
   
   //! Initiation by parametr container
   bool Init( const tnlGridOld< 2, Real, Device, Index >& u,
              const tnlParameterContainer& parameters,
              Index _overlap_width,
              int root = 0,
              MPI_Comm comm = MPI_COMM_WORLD );

   MPI_Comm GetMeshComm() const
   {
      return mesh_comm;
   }

   //! Get number of all nodes of the mesh
   int GetMeshSize() const
   {
      return mesh_x_size * mesh_y_size;
   };

   //! Get number of all nodes along x axis
   int GetXSize() const
   {
      return mesh_x_size;
   };

   Index GetSubdomainXSize() const
   {
      return subdomain_x_size;
   };

   Index GetSubdomainYSize() const
   {
      return subdomain_y_size;
   };

   //! Get number of all nodes along y axis
   int GetYSize() const
   {
      return mesh_y_size;
   };

   //! Get node x position
   int GetXPos() const
   {
      return node_x_pos;
   };

   //! Get node y position
   int GetYPos() const
   {
      return node_y_pos;
   };

   //! Get left neighbour of this node
   bool GetLeftNeighbour( int& nb_rank ) const
   {
      return left_neighbour;
   };

   //! Get right neighbour of this node
   bool GetRightNeighbour( int& nb_rank ) const
   {
      return right_neighbour;
   };

   //! Get lower neighbour of this node
   bool GetBottomNeighbour( int& nb_rank ) const
   {
      return bottom_neighbour;
   };

   //! Get upper neighbour of this node
   bool GetTopNeighbour( int& nb_rank ) const
   {
      return top_neighbour;
   };

   Index GetLeftOverlap() const
   {
      return left_overlap;
   };

   Index GetRightOverlap() const
   {
      return right_overlap;
   };

   Index GetBottomOverlap() const
   {
      return bottom_overlap;
   };

   Index GetTopOverlap() const
   {
      return top_overlap;
   };

   bool SetGlobalDomain( tnlGridOld< 2, Real, Device, Index >& global_u );

   //! Create subdomains
   bool CreateMesh( const tnlGridOld< 2, Real, Device, Index >& u,
                    tnlGridOld< 2, Real, Device, Index >& sub_u,
                    int root = 0 ) const;

   //! Scatter the function
   void Scatter( const tnlGridOld< 2, Real, Device, Index >& u,
                 tnlGridOld< 2, Real, Device, Index >& sub_u,
                 int root = 0 ) const;

   //! Scatter the function but only at the domains at the boundaries
   //void ScatterAtBoundaries( const tnlGridOld2D* u,
   //                          tnlGridOld2D* sub_u ); 

   //! Gather the function
   void Gather( tnlGridOld< 2, Real, Device, Index >& u,
                const tnlGridOld< 2, Real, Device, Index >& sub_u,
                int root = 0 ) const;

   //! Synchronize domain edges
   void Synchronize( tnlGridOld< 2, Real, Device, Index >& u );
   
   //! Get domain edges
   void DomainOverlaps( int& right, int& left,
                        int& bottom, int& top );

   protected:
   //! Supporting method for scattering
   void ScatterToNode( const tnlGridOld< 2, Real, Device, Index >& u,
                       tnlGridOld< 2, Real, Device, Index >& sub_u,
                       int dest_node,
                       int root ) const;

   //! Freeing momery used by buffers
   void FreeBuffers();

   protected:
   //! MPI communication group of the mesh
   MPI_Comm mesh_comm, original_comm;

   //! Mesh dimensions
   int mesh_x_size, mesh_y_size;

   //! Node coordinates
   int node_x_pos, node_y_pos;

   //! The neighbhors positions
   int left_neighbour, right_neighbour, bottom_neighbour, top_neighbour,
       left_bottom_neighbour, right_bottom_neighbour,
       left_top_neighbour, right_top_neighbour;
   
   //! Global domain dimensions
   Index domain_x_size, domain_y_size;

   //! Global domain size
   Real Ax, Bx, Ay, By;
    
   //! The subdomain dimensions
   Index subdomain_x_size, subdomain_y_size;
    
   //! The domain overlaps
   Index overlap_width, left_overlap, right_overlap, bottom_overlap, top_overlap;

   //! Buffers for MPI communication
   Real *left_send_buff, *right_send_buff,
        *bottom_send_buff, *top_send_buff,
        *bottom_left_send_buff,
        *bottom_right_send_buff,
        *top_left_send_buff,
        *top_right_send_buff,
        *left_recieve_buff, *right_recieve_buff,
        *bottom_recieve_buff, *top_recieve_buff,
        *bottom_left_recieve_buff,
        *bottom_right_recieve_buff,
        *top_left_recieve_buff,
        *top_right_recieve_buff;
};
               
template< typename Real, typename Device, typename Index >
void DrawSubdomains( const tnlMPIMesh< 2, Real, Device, Index >& mpi_mesh,
                     const tnlGridOld< 2, Real, Device, Index >* u,
                     const char* file_name_base,
                     const char* format );

// Implementation

template< typename Real, typename Device, typename Index >
tnlMPIMesh< 2, Real, Device, Index > :: tnlMPIMesh()
    : mesh_comm( 0 ), original_comm( 0 ),
      mesh_x_size( 0 ), mesh_y_size( 0 ),
      node_x_pos( 0 ) , node_y_pos( 0 ),
      left_neighbour( 0 ), right_neighbour( 0 ), bottom_neighbour( 0 ), top_neighbour( 0 ),
      left_bottom_neighbour( 0 ), right_bottom_neighbour( 0 ),
      left_top_neighbour( 0 ), right_top_neighbour( 0 ),
      domain_x_size( 0 ), domain_y_size( 0 ),
      Ax( 0 ), Bx( 0 ), Ay( 0 ), By( 0 ),
      subdomain_x_size( 0 ), subdomain_y_size( 0 ), overlap_width( 0 ),
      left_overlap( 0 ), right_overlap( 0 ), bottom_overlap( 0 ), top_overlap( 0 ),
      left_send_buff( 0 ), right_send_buff( 0 ),
      bottom_send_buff( 0 ), top_send_buff( 0 ),
      bottom_left_send_buff( 0 ),
      bottom_right_send_buff( 0 ),
      top_left_send_buff( 0 ),
      top_right_send_buff( 0 ),
      left_recieve_buff( 0 ), right_recieve_buff( 0 ),
      bottom_recieve_buff( 0 ), top_recieve_buff( 0 ),
      bottom_left_recieve_buff( 0 ),
      bottom_right_recieve_buff( 0 ),
      top_left_recieve_buff( 0 ),
      top_right_recieve_buff( 0 )
      {};
   
template< typename Real, typename Device, typename Index  >
bool tnlMPIMesh< 2, Real, Device, Index > :: Init( const tnlGridOld< 2, Real, Device, Index >& u,
                                                   int& _mesh_x_size,
                                                   int& _mesh_y_size,
                                                   Index _overlap_width,
                                                   int root,
                                                   MPI_Comm comm )
   {
#ifdef HAVE_MPI
      original_comm = comm;
      overlap_width = _overlap_width;
      mesh_x_size = _mesh_x_size;
      mesh_y_size = _mesh_y_size;
      :: MPIBcast< int >( mesh_x_size, 1, 0 );
      :: MPIBcast< int >( mesh_y_size, 1, 0 );
      int dims[ 2 ];
      dims[ 0 ] = mesh_x_size;
      dims[ 1 ] = mesh_y_size;
      if( ! mesh_x_size || ! mesh_y_size )
      {
         MPI_Dims_create( MPIGetSize( comm ), 2, dims );
         mesh_x_size = dims[ 0 ];
         mesh_y_size = dims[ 1 ];
      }
      int periods[ 2 ] = { false, false };
      MPI_Cart_create( comm, 2, dims, periods, true, &mesh_comm );
      int topo_type;
      MPI_Topo_test( mesh_comm, &topo_type );
      if( mesh_comm == MPI_COMM_NULL )
      {
         if( MPIGetRank( comm ) == root )
            cerr << "Not enough nodes for creating mesh " << mesh_x_size <<
                    "x" << mesh_y_size << endl;
         return false;
      }
      MPI_Cart_coords( mesh_comm, MPIGetRank( mesh_comm ), 2, dims );
      node_x_pos = dims[ 0 ];  
      node_y_pos = dims[ 1 ];  
      
      MPI_Cart_shift( mesh_comm, 0, 1, &left_neighbour, &right_neighbour );
      MPI_Cart_shift( mesh_comm, 1, 1, &bottom_neighbour, &top_neighbour );
      :: MPIBcast< int >( _overlap_width, 1, root );
      if( left_neighbour != MPI_PROC_NULL ) left_overlap = _overlap_width;
      else left_overlap = 0;
      if( right_neighbour != MPI_PROC_NULL ) right_overlap = _overlap_width;
      else right_overlap = 0;
      if( bottom_neighbour != MPI_PROC_NULL ) bottom_overlap = _overlap_width;
      else bottom_overlap = 0;
      if( top_neighbour != MPI_PROC_NULL ) top_overlap = _overlap_width;
      else top_overlap = 0;
      int coords[ 2 ];
      if( node_x_pos < mesh_x_size - 1 && node_y_pos < mesh_y_size - 1 )
      {
         coords[ 0 ] = node_x_pos + 1;
         coords[ 1 ] = node_y_pos + 1;
         MPI_Cart_rank( mesh_comm, coords, &right_top_neighbour );
      }
      else right_top_neighbour = MPI_PROC_NULL;
      if( node_x_pos > 0 && node_y_pos < mesh_y_size - 1 )
      {
         coords[ 0 ] = node_x_pos - 1;
         coords[ 1 ] = node_y_pos + 1;
         MPI_Cart_rank( mesh_comm, coords, &left_top_neighbour );
      }
      else left_top_neighbour = MPI_PROC_NULL;
      if( node_x_pos < mesh_x_size - 1 && node_y_pos > 0 )
      {
         coords[ 0 ] = node_x_pos + 1;
         coords[ 1 ] = node_y_pos - 1;
         MPI_Cart_rank( mesh_comm, coords, &right_bottom_neighbour );
      }
      else right_bottom_neighbour = MPI_PROC_NULL;
      if( node_x_pos > 0  && node_y_pos > 0  )
      {
         coords[ 0 ] = node_x_pos - 1;
         coords[ 1 ] = node_y_pos - 1;
         MPI_Cart_rank( mesh_comm, coords, &left_bottom_neighbour );
      }
      else left_bottom_neighbour = MPI_PROC_NULL;

      domain_x_size = u. GetXSize();
      domain_y_size = u. GetYSize();
      :: MPIBcast< int >( domain_x_size, 1, 0 );
      :: MPIBcast< int >( domain_y_size, 1, 0 );
      
      subdomain_x_size = domain_x_size / mesh_x_size;
      subdomain_y_size = domain_y_size / mesh_y_size;
      
      if( node_x_pos == mesh_x_size - 1 )
         subdomain_x_size = domain_x_size - subdomain_x_size * ( mesh_x_size - 1 );
      
      if( node_y_pos == mesh_y_size - 1 )
         subdomain_y_size = domain_y_size - subdomain_y_size * ( mesh_y_size - 1 );
      
      Ax = u. GetAx();
      Ay = u. GetAy();
      Bx = u. GetBx();
      By = u. GetBy();
      
      :: MPIBcast< double >( Ax, 1, 0 );
      :: MPIBcast< double >( Ay, 1, 0 );
      :: MPIBcast< double >( Bx, 1, 0 );
      :: MPIBcast< double >( By, 1, 0 );

      FreeBuffers();
      if( left_overlap )
      {
         left_send_buff = new T[ subdomain_y_size * left_overlap ];
         left_recieve_buff = new T[ subdomain_y_size * left_overlap ];
      }
      if( right_overlap )
      {
         right_send_buff = new T[ subdomain_y_size * right_overlap ];
         right_recieve_buff = new T[ subdomain_y_size * right_overlap ];
      }
      if( bottom_overlap )
      {
         bottom_send_buff = new T[ subdomain_x_size * bottom_overlap ];
         bottom_recieve_buff = new T[ subdomain_x_size * bottom_overlap ];
      }
      if( top_overlap )
      {
         top_send_buff = new T[ subdomain_x_size * top_overlap ];
         top_recieve_buff = new T[ subdomain_x_size * top_overlap ];
      }
      if( bottom_overlap && left_overlap )
      {
         bottom_left_send_buff = new T[ bottom_overlap * left_overlap ];
         bottom_left_recieve_buff = new T[ bottom_overlap * left_overlap ];
      }
   
      if( bottom_overlap && right_overlap )
      {
         bottom_right_send_buff = new T[ bottom_overlap * right_overlap ];
         bottom_right_recieve_buff = new T[ bottom_overlap * right_overlap ];
      }
      if( top_overlap && left_overlap )
      {
         top_left_send_buff = new T[ top_overlap * left_overlap ];
         top_left_recieve_buff = new T[ top_overlap * left_overlap ];
      }
      if( top_overlap && right_overlap )
      {
         top_right_send_buff = new T[ top_overlap * right_overlap ];
         top_right_recieve_buff = new T[ top_overlap * right_overlap ];
      }
      cout << "Node " << MPIGetRank() 
           << " has position (" << GetXPos() 
           << ", " << GetYPos() 
           << ") and dimensions " << GetSubdomainXSize() 
           << " x " << GetSubdomainYSize() << endl;
#else
      domain_x_size = u. getDimensions(). x();
      domain_y_size = u. getDimensions(). y();
      
      Ax = u. getDomainLowerCorner(). x();
      Ay = u. getDomainLowerCorner(). y();
      Bx = u. getDomainUpperCorner(). x();
      By = u. getDomainUpperCorner(). y();

#endif
      return true;
   };
   
template< typename Real, typename Device, typename Index  >
bool tnlMPIMesh< 2, Real, Device, Index > :: Init( const tnlGridOld< 2, Real, Device, Index >& u,
                                                   const tnlParameterContainer& parameters,
                                                   Index _overlap_width,
                                                   int root,
                                                   MPI_Comm comm )
{
   MPIBarrier();
   int mpi_mesh_x_size = parameters. GetParameter< int >( "mpi-mesh-x-size" );
   int mpi_mesh_y_size = parameters. GetParameter< int >( "mpi-mesh-y-size" );
   return Init( u, mpi_mesh_x_size, mpi_mesh_y_size, _overlap_width, root, comm );
}

template< typename Real, typename Device, typename Index  >
bool tnlMPIMesh< 2, Real, Device, Index > :: SetGlobalDomain( tnlGridOld< 2, Real, Device, Index >& global_u )
{
   if( ! global_u. setDimensions( tnlTuple< 2, int >( domain_x_size, domain_y_size ) ) )
      return false;
   
   global_u. setDomain( tnlTuple< 2, Real >( Ax, Ay ),
                        tnlTuple< 2, Real >( Bx, By ) );
   return true;
}
   
template< typename Real, typename Device, typename Index  >
bool tnlMPIMesh< 2, Real, Device, Index > :: CreateMesh( const tnlGridOld< 2, Real, Device, Index >& u,
                                                         tnlGridOld< 2, Real, Device, Index >& sub_u,
                                                         int root ) const
{
#ifdef HAVE_MPI
   double ax, ay, hx, hy;
   tnlString name;
   int rank;
   if( MPIGetRank( original_comm ) == root )
   {
      ax = u. GetAx();
      ay = u. GetAy();
      hx = u. GetHx();
      hy = u. GetHy();
      name. setString( u. getName(). getString() );  
   }
   :: MPIBcast< double >( ax, 1, root, original_comm );
   :: MPIBcast< double >( ay, 1, root, original_comm );
   :: MPIBcast< double >( hx, 1, root, original_comm );
   :: MPIBcast< double >( hy, 1, root, original_comm );
   name. MPIBcast( root, original_comm );
   int err( 0 ), all_err( 0 );
   if( ! sub_u. SetNewDimensions( subdomain_x_size + left_overlap + right_overlap,
                                  subdomain_y_size + bottom_overlap + top_overlap ) )
   {
      cerr << "Unable to allocate subdomain grids for '" << name 
           << "' on the node ( " << node_x_pos << ", " << node_y_pos 
           << " rank " << MPIGetRank( original_comm ) << "." << endl;
      err = 1;
   }
   sub_u. SetNewDomain( ax + ( node_x_pos * subdomain_x_size - left_overlap ) * hx,
                        ax + ( ( node_x_pos + 1 ) * subdomain_x_size + right_overlap - 1 ) * hx,
                        ay + ( node_y_pos * subdomain_y_size - bottom_overlap ) * hx,
                        ay + ( ( node_y_pos + 1 ) * subdomain_y_size + top_overlap - 1 ) * hx,
                        hx, hy );
   //cout << "Node " << MPIGetRank() << " mesh size " 
   //     << sub_u -> GetXSize() << "x" << sub_u -> GetYSize() << endl;
   sub_u. setName( name. getString() );
   MPI_Allreduce( &err, &all_err, 1, MPI_INT,MPI_SUM, mesh_comm );
   if( all_err != 0 ) return false;
#else
   sub_u. setLike( u );
#endif
return true;
};

template< typename Real, typename Device, typename Index  >
void tnlMPIMesh< 2, Real, Device, Index > :: ScatterToNode( const tnlGridOld< 2, Real, Device, Index >& u,
                                                            tnlGridOld< 2, Real, Device, Index >& sub_u,
                                                            int dest_node,
                                                            int root ) const
{
   dbgFunctionName( "tnlMPIMesh", "ScatterToNode" );
#ifdef HAVE_MPI
   if( MPIGetRank( original_comm ) == root )
   {
      //cout << "Node " << MPIGetRank() << " scatter to " << dest_node << endl;
      int dest_x_pos;
      int dest_y_pos;
      int coords[ 2 ];
      MPI_Cart_coords( mesh_comm, dest_node, 2, coords );
      dest_x_pos = coords[ 0 ];
      dest_y_pos = coords[ 1 ];

      int dest_left_overlap( 0 ), dest_right_overlap( 0 ),
          dest_bottom_overlap( 0 ), dest_top_overlap( 0 );
      dbgExpr( dest_node );
      if( dest_x_pos > 0 ) dest_left_overlap = overlap_width;
      if( dest_x_pos < mesh_x_size - 1 ) dest_right_overlap = overlap_width;
      if( dest_y_pos > 0 ) dest_bottom_overlap = overlap_width;
      if( dest_y_pos < mesh_y_size - 1 ) dest_top_overlap = overlap_width;
     
      dbgCout( "dest edges LRLU " << 
               dest_left_overlap << dest_right_overlap <<
               dest_bottom_overlap << dest_top_overlap );

      tnlGridOld< 2, Real, Device, Index >* mpi_buff;
      if( dest_node == root )
         mpi_buff = &sub_u;
      else mpi_buff = new tnlGridOld< 2, Real, Device, Index > ( subdomain_x_size + dest_left_overlap + dest_right_overlap,
                                         subdomain_y_size + dest_bottom_overlap + dest_top_overlap,
                                         0.0, 1.0, 0.0, 1.0 );
      dbgCout( "mpi buff < " <<
                  dest_x_pos * subdomain_x_size - dest_left_overlap << ", " <<
                  ( dest_x_pos + 1 ) * subdomain_x_size + dest_right_overlap << " > x < " <<
                  dest_y_pos * subdomain_y_size - dest_bottom_overlap << ", " <<
                  ( dest_y_pos + 1 ) * subdomain_y_size + dest_top_overlap << " >" );
      
      int i, j;
      const int i1 = dest_x_pos * subdomain_x_size - dest_left_overlap;
      const int i2 = ( dest_x_pos + 1 ) * subdomain_x_size + dest_right_overlap;
      const int j1 = dest_y_pos * subdomain_y_size - dest_bottom_overlap;
      const int j2 = ( dest_y_pos + 1 ) * subdomain_y_size + dest_top_overlap;
      for( i = i1; i < i2; i ++ )
         for( j = j1; j < j2; j ++ )
            ( *mpi_buff )( i - i1, j - j1 ) = u( i, j );
      int buf_size = 
         ( subdomain_x_size + dest_left_overlap + dest_right_overlap ) *
         ( subdomain_y_size + dest_bottom_overlap + dest_top_overlap );
      if( dest_node != root )
      {
         MPI_Send( mpi_buff -> getData(),
                   buf_size * sizeof( T ),
                   MPI_BYTE,
                   dest_node,
                   0,
                   mesh_comm );
         delete mpi_buff;
      }
      return;
   }
   if( dest_node == root ) return;
   int buf_size =
         ( subdomain_x_size + left_overlap + right_overlap ) *
         ( subdomain_y_size + bottom_overlap + top_overlap );
   MPI_Status status;
   MPI_Recv( sub_u. getData(),
             buf_size * sizeof( T ),
             MPI_BYTE,
             0,
             0,
             mesh_comm,
             &status );
   //cout << "Receiving data on node " << node_rank << endl;
#endif
}

template< typename Real, typename Device, typename Index  >
void tnlMPIMesh< 2, Real, Device, Index > :: Scatter( const tnlGridOld< 2, Real, Device, Index >& u,
                                                      tnlGridOld< 2, Real, Device, Index >& sub_u,
                                                      int root ) const
{
#ifdef HAVE_MPI
   if( MPIGetRank( original_comm ) == root )
   {
      int dest, mesh_size = mesh_x_size * mesh_y_size;
      for( dest = 0; dest < mesh_size; dest ++ )
         ScatterToNode( u, sub_u, dest, root );
   }
   else ScatterToNode( u, sub_u, MPI_PROC_NULL, root );
#else
   if( &u == &sub_u ) return;
   sub_u = u;
#endif
}
    
template< typename Real, typename Device, typename Index  >
void tnlMPIMesh< 2, Real, Device, Index > :: Gather( tnlGridOld< 2, Real, Device, Index >& u,
                                                     const tnlGridOld< 2, Real, Device, Index >& sub_u,
                                                     int root ) const
{
   dbgFunctionName( "tnlMPIMesh", "Gather" );
#ifdef HAVE_MPI
   
   dbgMPIBarrier;
   dbgExpr( MPIGetRank( original_comm ) );
   if( MPIGetRank( original_comm ) == root )
   {
      int src, mesh_size = mesh_x_size * mesh_y_size;
      int i, j;
      MPI_Status status;
      for( src = 0; src < mesh_size; src ++ )
      {
         int coords[ 2 ];
         MPI_Cart_coords( mesh_comm, src, 2, coords ); 
         int src_x_pos = coords[ 0 ];
         int src_y_pos = coords[ 1 ];

         dbgExpr( src );
         dbgExpr( src_x_pos );
         dbgExpr( src_y_pos );

         int src_left_overlap( 0 ), src_right_overlap( 0 ),
             src_bottom_overlap( 0 ), src_top_overlap( 0 );
         if( src_x_pos > 0 ) src_left_overlap = overlap_width;
         if( src_x_pos < mesh_x_size - 1 ) src_right_overlap = overlap_width;
         if( src_y_pos > 0 ) src_bottom_overlap = overlap_width;
         if( src_y_pos < mesh_y_size - 1 ) src_top_overlap = overlap_width;
         if( src != root )
         {
            
            dbgCout( "Allocating supporting buffer < " <<
                      src_x_pos * subdomain_x_size - src_left_overlap <<
                      ", " << ( src_x_pos + 1 ) * subdomain_x_size + src_right_overlap <<
                      " >x< " << src_y_pos * subdomain_y_size - src_bottom_overlap <<
                      ", " << ( src_y_pos + 1 ) * subdomain_y_size + src_top_overlap <<
                      " >" );
                  
            tnlGridOld< 2, Real, Device, Index > mpi_buff( subdomain_x_size + src_left_overlap + src_right_overlap,
                                   subdomain_y_size + src_bottom_overlap + src_top_overlap,
                                   0.0, 1.0, 0.0, 1.0 );
            int buf_size = 
               ( subdomain_x_size + src_left_overlap + src_right_overlap ) *
               ( subdomain_y_size + src_bottom_overlap + src_top_overlap );
            dbgExpr( buf_size );
            
            dbgCout( "RECEIVING data from node " << src  );
            MPI_Recv( mpi_buff. getData(),
                      buf_size * sizeof( T ),
                      MPI_BYTE,
                      src,
                      0,
                      mesh_comm,
                      &status );
            dbgCout( "Receiving data done." );
            const int i1 = src_x_pos * subdomain_x_size;
            const int i2 = i1 + subdomain_x_size;
            const int j1 = src_y_pos * subdomain_y_size;
            const int j2 = j1 + subdomain_y_size;
            for( i = i1; i < i2; i ++ )
               for( j = j1; j < j2; j ++ )
               {
                  //cout << "Node recv" << MPIGetRank( original_comm ) << " i = " << i << " j = " << j << endl;
                  u( i, j ) = mpi_buff( i - i1 + src_left_overlap, j - j1 + src_bottom_overlap );
               }
         }
         else
         {
            const int i1 = src_x_pos * subdomain_x_size;
            const int i2 = i1 + subdomain_x_size;
            const int j1 = src_y_pos * subdomain_y_size;
            const int j2 = j1 + subdomain_y_size;
            for( i = i1; i < i2; i ++ )
               for( j = j1; j < j2; j ++ )
               {
                  //cout << "Node cp" << MPIGetRank( original_comm ) << " i = " << i << " j = " << j << endl;
                  u( i, j ) = sub_u( i - i1 + src_left_overlap, j - j1 + bottom_overlap );
               }
         }
      }
   }
   else
   {
      dbgCout( "node rank " << MPIGetRank( original_comm ) << " SENDING data" );
      int buf_size = ( subdomain_x_size + left_overlap + right_overlap ) *
                          ( subdomain_y_size + bottom_overlap + top_overlap );
      MPI_Send( const_cast< T* >( sub_u. getData() ),
                buf_size * sizeof( T ),
                MPI_BYTE,
                root,
                0,
                mesh_comm );
      dbgCout( "sending data done." );
   }
   dbgCout( "Gathering data done." );
#else
   if( &u == &sub_u ) return;
   u = sub_u;
#endif
}

template< typename Real, typename Device, typename Index  >
void tnlMPIMesh< 2, Real, Device, Index > :: Synchronize( tnlGridOld< 2, Real, Device, Index >& u )
{
   dbgFunctionName( "tnlMPIMesh", "Synchronize" );
#ifdef HAVE_MPI
  int i, j;
  int min_x = left_overlap;
  int min_y = bottom_overlap;
  int max_x = min_x + subdomain_x_size;
  int max_y = min_y + subdomain_y_size;
  int wdth = overlap_width;
  MPI_Status status;
   
  MPI_Request lft_snd_rqst, rght_snd_rqst, lwr_snd_rqst, uppr_snd_rqst,
              lwr_lft_snd_rqst, lwr_rght_snd_rqst,
              uppr_lft_snd_rqst, uppr_rght_snd_rqst,
              lft_rcv_rqst, rght_rcv_rqst, lwr_rcv_rqst, uppr_rcv_rqst,
              lwr_lft_rcv_rqst, lwr_rght_rcv_rqst,
              uppr_lft_rcv_rqst, uppr_rght_rcv_rqst;
  
  
  dbgMPIBarrier;
  
  //starting communication with the left neighbour
  if( left_neighbour != MPI_PROC_NULL )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - SENDING data to the LEFT neighbour" );
     for( i = 0; i < wdth; i ++ )
        for( j = 0; j < subdomain_y_size; j ++ )
           left_send_buff[ i * subdomain_y_size + j ] =
              u( i + min_x, j + min_y );
     MPI_Isend( left_send_buff,
                wdth * subdomain_y_size * sizeof( T ),
                MPI_BYTE,
                left_neighbour,
                0,
                mesh_comm ,
                &lft_snd_rqst );
     
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - RECEIVING data from the LEFT neighbour" );
     MPI_Irecv( left_recieve_buff,
                wdth * subdomain_y_size * sizeof( T ),
                MPI_CHAR,
                left_neighbour,
                0,
                mesh_comm,
                &lft_rcv_rqst );
  }
  dbgMPIBarrier;
  
  // starting communication with the right neighbour
  if( right_neighbour != MPI_PROC_NULL )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - SENDING data to the RIGHT neighbour" );
     for( i = 0; i < wdth; i ++ )
        for( j = 0; j < subdomain_y_size; j ++ )
           right_send_buff[ i * subdomain_y_size + j ] =
               u( max_x - wdth + i, j + min_y );
     MPI_Isend( right_send_buff,
                wdth * subdomain_y_size * sizeof( T ),
                MPI_BYTE,
                right_neighbour,
                0,
                mesh_comm,
                &rght_snd_rqst );
     
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - RECEIVING data from the RIGHT neighbour" );
     MPI_Irecv( right_recieve_buff,
                wdth * subdomain_y_size * sizeof( T ),
                MPI_BYTE,
                right_neighbour,
                0,
                mesh_comm,
                &rght_rcv_rqst );
  }
  dbgMPIBarrier;
  
  // starting communication with the bottom neighbour
  if( bottom_neighbour != MPI_PROC_NULL )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - SENDING data to the BOTTOM neighbour" );
     for( i = 0; i < subdomain_x_size; i ++ )
        for( j = 0; j < wdth; j ++ )
           bottom_send_buff[ j * subdomain_x_size + i ] =
              u( min_x + i, min_y + j);
     MPI_Isend( bottom_send_buff,
                wdth * subdomain_x_size * sizeof( T ),
                MPI_BYTE,
                bottom_neighbour,
                0,
                mesh_comm,
                &lwr_snd_rqst );
     
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - RECEIVING data from the BOTTOM neighbour" );
     MPI_Irecv( bottom_recieve_buff,
                wdth * subdomain_y_size * sizeof( T ),
                MPI_BYTE,
                bottom_neighbour,
                0,
                mesh_comm,
                &lwr_rcv_rqst );
  }
  dbgMPIBarrier;

  // starting communication with the uppper neighbour
  if( top_neighbour != MPI_PROC_NULL )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - SENDING data to the TOP neighbour" );
     for( i = 0; i < subdomain_x_size; i ++ )
        for( j = 0; j < wdth; j ++ )
           top_send_buff[ j * subdomain_x_size + i ] =
             u( min_x + i, max_y - wdth + j);
     MPI_Isend( top_send_buff,
                wdth * subdomain_x_size * sizeof( T ),
                MPI_BYTE,
                top_neighbour,
                0,
                mesh_comm,
                &uppr_snd_rqst );
     
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - RECEIVING data from the TOP neighbour" );
     MPI_Irecv( top_recieve_buff,
                wdth * subdomain_y_size * sizeof( T ),
                MPI_BYTE,
                top_neighbour,
                0,
                mesh_comm,
                &uppr_rcv_rqst );
  }
  dbgMPIBarrier;
  
  int wdth_2 = wdth * wdth;
  
  // starting communication with lower left neighbour
  if( left_bottom_neighbour != MPI_PROC_NULL )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - SENDING small square to the BOTTOM LEFT neighbour." );
     for( i = 0; i < wdth; i ++ )
        for( j = 0; j < wdth; j ++ )
           bottom_left_send_buff[ j * wdth + i ] =
              u( min_x + i, min_y + j );
     MPI_Isend( bottom_left_send_buff,
                wdth_2 * sizeof( T ),
                MPI_BYTE,
                left_bottom_neighbour,
                0,
                mesh_comm,
                &lwr_lft_snd_rqst );
     
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - RECEIVING small square from the BOTTOM LEFT neighbour." );
     MPI_Irecv( bottom_left_recieve_buff,
                wdth_2 * sizeof( T ),
                MPI_BYTE,
                left_bottom_neighbour,
                0,
                mesh_comm,
                &lwr_lft_rcv_rqst );
  }
  dbgMPIBarrier;
  
  // starting communication with lower right neighbour
  if( right_bottom_neighbour != MPI_PROC_NULL )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - SENDING small square to the BOTTOM RIGHT neighbour." );
     for( i = 0; i < wdth; i ++ )
        for( j = 0; j < wdth; j ++ )
           bottom_right_send_buff[ j * wdth + i ] =
              u( max_x - wdth + i, min_y + j );
     MPI_Isend( bottom_right_send_buff,
                wdth_2 * sizeof( T ),
                MPI_BYTE,
                right_bottom_neighbour,
                0,
                mesh_comm,
                &lwr_rght_snd_rqst );
     
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - RECEIVING small square from the BOTTOM RIGHT neighbour." );
     MPI_Irecv( bottom_right_recieve_buff,
                wdth_2 * sizeof( T ),
                MPI_BYTE,
                right_bottom_neighbour,
                0,
                mesh_comm,
                &lwr_rght_rcv_rqst );
  }
  dbgMPIBarrier;

  // starting communication with upper left neighbour
  if( left_top_neighbour != MPI_PROC_NULL )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - SENDING small square to the TOP LEFT neighbour." );
     for( i = 0; i < wdth; i ++ )
        for( j = 0; j < wdth; j ++ )
           top_left_send_buff[ j * wdth + i ] =
              u( min_x + i, max_y - wdth + j );
     MPI_Isend( top_left_send_buff,
                wdth_2 * sizeof( T ),
                MPI_BYTE,
                left_top_neighbour,
                0,
                mesh_comm,
                &uppr_lft_snd_rqst );
     
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - RECEIVING small square from the TOP LEFT neighbour." );
     MPI_Irecv( top_left_recieve_buff,
                wdth_2 * sizeof( T ),
                MPI_BYTE,
                left_top_neighbour,
                0,
                mesh_comm,
                &uppr_lft_rcv_rqst );
  }
  dbgMPIBarrier;

  // starting communication with upper right neighbour
  if( right_top_neighbour != MPI_PROC_NULL )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - SENDING small square to the TOP RIGHT neighbour." );
     for( i = 0; i < wdth; i ++ )
        for( j = 0; j < wdth; j ++ )
           top_right_send_buff[ j * wdth + i ] =
              u( max_x - wdth + i, max_y - wdth + j );
     MPI_Isend( top_right_send_buff,
                wdth_2 * sizeof( T ),
                MPI_BYTE,
                right_top_neighbour,
                0,
                mesh_comm,
                &uppr_rght_snd_rqst );
     
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - RECEIVING small square from the TOP RIGHT neighbour." );
     MPI_Irecv( top_right_recieve_buff,
                wdth_2 * sizeof( T ),
                MPI_BYTE,
                right_top_neighbour,
                0,
                mesh_comm,
                &uppr_rght_rcv_rqst );
  }
  dbgMPIBarrier;

  // finishing communication with the left neighbour
  if( left_neighbour != MPI_PROC_NULL )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - WAITING for data from LEFT neighbour." );
     MPI_Wait( &lft_rcv_rqst, &status );
     for( i = 0; i < wdth; i ++ )
        for( j = 0; j < subdomain_y_size; j ++ )
           u( min_x - wdth + i, min_y + j ) =
           left_recieve_buff[ i * subdomain_y_size + j ];
     MPI_Wait( &lft_snd_rqst, &status );
  }
  dbgMPIBarrier;
  
  // finishing communication with the right neighbour
  if( right_neighbour != MPI_PROC_NULL )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - WAITING for data from RIGHT neighbour." );
     MPI_Wait( &rght_rcv_rqst, &status );
     for( i = 0; i < wdth; i ++ )
        for( j = 0; j < subdomain_y_size; j ++ )
           u( max_x + i, min_y + j ) =
           right_recieve_buff[ i * subdomain_y_size + j ];
     MPI_Wait( &rght_snd_rqst, &status );
  }
  dbgMPIBarrier;
  
  // finishing communication with the lower neighbour
  if( bottom_neighbour != MPI_PROC_NULL )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - WAITING for data from BOTTOM neighbour." );
     MPI_Wait( &lwr_rcv_rqst, &status );
     for( i = 0; i < subdomain_x_size; i ++ )
        for( j = 0; j < wdth; j ++ )
           u( min_x + i, min_y - wdth + j ) =
           bottom_recieve_buff[ j * subdomain_x_size + i ];
     MPI_Wait( &lwr_snd_rqst, &status );
  }
  dbgMPIBarrier;
  
  // finishing communication with the upper neighbour
  if( top_neighbour != MPI_PROC_NULL )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - WAITING for data from TOP neighbour." );
     MPI_Wait( &uppr_rcv_rqst, &status );
     for( i = 0; i < subdomain_x_size; i ++ )
        for( j = 0; j < wdth; j ++ )
           u( min_x + i, max_y + j ) =
           top_recieve_buff[ j * subdomain_x_size + i ];
     MPI_Wait( &uppr_snd_rqst, &status );
  }
  dbgMPIBarrier;
  
  
  // finishing communication with the lower left neighbour
  if( left_bottom_neighbour != MPI_PROC_NULL  )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - WAITING for data from BOTTOM LEFT neighbour." );
     MPI_Wait( &lwr_lft_rcv_rqst, &status );
     for( i = 0; i < wdth; i ++ )
        for( j = 0; j < wdth; j ++ )
           u( min_x - wdth + i, min_y - wdth + j ) =
           bottom_left_recieve_buff[ j * wdth + i ];
     MPI_Wait( &lwr_lft_snd_rqst, &status );
  }
  dbgMPIBarrier;
 
  // finishing communication with the lower right neighbour
  if( right_bottom_neighbour != MPI_PROC_NULL )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - WAITING for data from BOTTOM RIGHT neighbour." );
     MPI_Wait( &lwr_rght_rcv_rqst, &status );
     for( i = 0; i < wdth; i ++ )
        for( j = 0; j < wdth; j ++ )
           u( max_x + i, min_y - wdth + j ) =
           bottom_right_recieve_buff[ j * wdth + i ];
     MPI_Wait( &lwr_rght_snd_rqst, &status );
  }
  dbgMPIBarrier;

  // finishing communication with the upper right neighbour
  if( right_top_neighbour != MPI_PROC_NULL )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - WAITING for data from TOP RIGHT neighbour." );
     MPI_Wait( &uppr_rght_rcv_rqst, &status );
     for( i = 0; i < wdth; i ++ )
        for( j = 0; j < wdth; j ++ )
           u( max_x + i, max_y + j ) =
           top_right_recieve_buff[ j * wdth + i ];
     MPI_Wait( &uppr_rght_snd_rqst, &status );
  }
  dbgMPIBarrier;
  
  // finishing communication with the upper left neighbour
  if( left_top_neighbour != MPI_PROC_NULL )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - WAITING for data from TOP LEFT neighbour." );
     MPI_Wait( &uppr_lft_rcv_rqst, &status );
     for( i = 0; i < wdth; i ++ )
        for( j = 0; j < wdth; j ++ )
           u( min_x - wdth + i, max_y + j ) =
           top_left_recieve_buff[ j * wdth + i ];
     MPI_Wait( &uppr_lft_snd_rqst, &status );
  }
  
  dbgCout( "Synchronisation done..." );
  dbgMPIBarrier;
#endif
}
template< typename Real, typename Device, typename Index  >
void tnlMPIMesh< 2, Real, Device, Index > :: FreeBuffers()
{
   if( left_send_buff ) delete left_send_buff;
   if( right_send_buff ) delete right_send_buff;
   if( bottom_send_buff ) delete bottom_send_buff;
   if( top_send_buff ) delete top_send_buff;
   if( bottom_left_send_buff ) delete bottom_left_send_buff;
   if( bottom_right_send_buff ) delete bottom_right_send_buff;
   if( top_left_send_buff ) delete top_left_send_buff;
   if( top_right_send_buff ) delete top_right_send_buff;
   if( left_recieve_buff ) delete left_recieve_buff;
   if( right_recieve_buff ) delete right_recieve_buff;
   if( bottom_recieve_buff ) delete bottom_recieve_buff;
   if( top_recieve_buff ) delete top_recieve_buff;
   if( bottom_left_recieve_buff ) delete bottom_left_recieve_buff;
   if( bottom_right_recieve_buff ) delete bottom_right_recieve_buff;
   if( top_left_recieve_buff ) delete top_left_recieve_buff;
   if( top_right_recieve_buff ) delete top_right_recieve_buff;
   left_send_buff = 0;
   right_send_buff = 0;
   bottom_send_buff = 0;
   top_send_buff = 0;
   bottom_left_send_buff = 0;
   bottom_right_send_buff = 0;
   top_left_send_buff = 0;
   top_right_send_buff = 0;
   left_recieve_buff = 0;
   right_recieve_buff = 0;
   bottom_recieve_buff = 0;
   top_recieve_buff = 0;
   bottom_left_recieve_buff = 0;
   bottom_right_recieve_buff = 0;
   top_left_recieve_buff = 0;
   top_right_recieve_buff = 0;
};

template< typename Real, typename Device, typename Index  >
void DrawSubdomains( const tnlMPIMesh< 2, Real, Device, Index >& mpi_mesh,
                     const tnlGridOld< 2, Real, Device, Index >& u,
                     const char* file_name_base,
                     const char* format )
{
   int num = mpi_mesh. GetXPos() * 10 + mpi_mesh. GetYPos();
   tnlString file_name;
   FileNameBaseNumberEnding( file_name_base,
                             num,
                             2,
                             0,
                             file_name );
   Draw( u, file_name. getString(), format );
};

#endif
