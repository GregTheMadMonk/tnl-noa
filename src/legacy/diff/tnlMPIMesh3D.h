/***************************************************************************
                          tnlMPIMesh3D.h  -  description
                             -------------------
    begin                : 2009/07/26
    copyright            : (C) 2009 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#ifndef tnlMPIMesh3DH
#define tnlMPIMesh3DH

#include <debug/tnlDebug.h>
#include <legacy/mesh/tnlGridOld.h>
#include <core/mpi-supp.h>
#include <debug/tnlDebug.h>

template< typename Real, typename Device, typename Index >
class tnlMPIMesh< 3, Real, Device, Index >
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
   bool Init( const tnlGridOld< 3, Real, Device, Index >& u,
              int& _mesh_x_size,
              int& _mesh_y_size,
              int& _mesh_z_size,
              Index _overlap_width,
              int root = 0,
              MPI_Comm comm = MPI_COMM_WORLD );

   bool Init( const tnlGridOld< 3, Real, Device, Index >& u,
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
      return mesh_x_size * mesh_y_size * mesh_z_size;
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

   //! Get number of all nodes along y axis
   int GetYSize() const
   {
      return mesh_y_size;
   };

   Index GetSubdomainYSize() const
   {
      return subdomain_y_size;
   };

   //! Get number of all nodes along z axis
   int GetZSize() const
   {
      return mesh_z_size;
   };

   Index GetSubdomainZSize() const
   {
      return subdomain_z_size;
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

   //! Get node z position
   int GetZPos() const
   {
      return node_z_pos;
   };

   //! Get left neighbor of this node
   bool GetLeftNeighbour( int& nb_rank ) const
   {
      return left_neighbour;
   };

   //! Get right neighbor of this node
   bool GetRightNeighbour( int& nb_rank ) const
   {
      return right_neighbour;
   };

   //! Get lower neighbor of this node
   bool GetBottomNeighbour( int& nb_rank ) const
   {
      return bottom_neighbour;
   };

   //! Get upper neighbor of this node
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

   Index GetCloserOverlap() const
   {
      return closer_overlap;
   };

   Index GetFurtherOverlap() const
   {
      return further_overlap;
   };

   bool SetGlobalDomain( tnlGridOld< 3, Real, Device, Index >& global_u )
   {
      if( ! global_u. setDimensions( tnlStaticVector< 3, Index >( domain_x_size, domain_y_size, domain_z_size  ) ) )
         return false;
 
      global_u. setDomain( tnlStaticVector< 3, Real >( Ax, Bx, Ay ),
                           tnlStaticVector< 3, Real >( By, Az, Bz ) );
      return true;
   }

   //! Create subdomains
   bool CreateMesh( const tnlGridOld< 3, Real, Device, Index >& u,
                    tnlGridOld< 3, Real, Device, Index >& sub_u,
                    int root = 0 ) const;

   //! Scatter the function
   void Scatter( const tnlGridOld< 3, Real, Device, Index >& u,
                 tnlGridOld< 3, Real, Device, Index >& sub_u,
                 int root = 0 ) const;

   //! Scatter the function but only at the domains at the boundaries
   //void ScatterAtBoundaries( const tnlGridOld2D* u,
   //                          tnlGridOld2D* sub_u );

   //! Gather the function
   void Gather( tnlGridOld< 3, Real, Device, Index >& u,
                const tnlGridOld< 3, Real, Device, Index >& sub_u,
                int root = 0 ) const;

   //! Synchronize domain edges
   void Synchronize( tnlGridOld< 3, Real, Device, Index >& u );
 
   //! Get domain edges
   void DomainOverlaps( int& right, int& left,
                        int& bottom, int& top );

   protected:
   //! Supporting method for scattering
   void ScatterToNode( const tnlGridOld< 3, Real, Device, Index >& u,
                       tnlGridOld< 3, Real, Device, Index >& sub_u,
                       int dest_node,
                       int root ) const;

   //! Freeing memory used by buffers
   void FreeBuffers();

   protected:
   //! MPI communication group of the mesh
   MPI_Comm mesh_comm, original_comm;

   //! Mesh dimensions
   int mesh_x_size, mesh_y_size, mesh_z_size;

   //! Node coordinates
   int node_x_pos, node_y_pos, node_z_pos;

   //! The neighbours positions
   int left_neighbour, right_neighbour,
       bottom_neighbour, top_neighbour,
       closer_neighbour, further_neighbour,
       left_bottom_neighbour, right_bottom_neighbour,
       left_top_neighbour, right_top_neighbour;
 
   //! The subdomain dimensions
   Index subdomain_x_size, subdomain_y_size, subdomain_z_size;
 
   //! Global domain dimensions
   Index domain_x_size, domain_y_size, domain_z_size;

   //! Global domain size
   Real Ax, Bx, Ay, By, Az, Bz;
 
   //! The domain overlaps
   Index overlap_width,
       left_overlap, right_overlap,
       bottom_overlap, top_overlap,
       closer_overlap, further_overlap;

   //! Buffers for MPI communication
   Real *left_send_buff, *right_send_buff,
        *bottom_send_buff, *top_send_buff,
        *closer_send_buff, *further_send_buff,
        *bottom_left_send_buff,
        *bottom_right_send_buff,
        *top_left_send_buff,
        *top_right_send_buff,
        *left_recieve_buff, *right_recieve_buff,
        *bottom_recieve_buff, *top_recieve_buff,
        *closer_recieve_buff, *further_recieve_buff,
        *bottom_left_recieve_buff,
        *bottom_right_recieve_buff,
        *top_left_recieve_buff,
        *top_right_recieve_buff;
};
 
template< typename Real, typename Device, typename Index >
void DrawSubdomains( const tnlMPIMesh< 3, Real, Device, Index >& mpi_mesh,
                                         const tnlGridOld< 3, Real, Device, Index >& u,
                                         const char* file_name_base,
                                         const char* format );

// Implementation

template< typename Real, typename Device, typename Index >
tnlMPIMesh< 3, Real, Device, Index > :: tnlMPIMesh()
    : mesh_comm( 0 ), original_comm( 0 ),
      mesh_x_size( 0 ), mesh_y_size( 0 ), mesh_z_size( 0 ),
      node_x_pos( 0 ) , node_y_pos( 0 ), node_z_pos( 0 ),
      left_neighbour( 0 ), right_neighbour( 0 ),
      bottom_neighbour( 0 ), top_neighbour( 0 ),
      closer_neighbour( 0 ), further_neighbour( 0 ),
      left_bottom_neighbour( 0 ), right_bottom_neighbour( 0 ),
      left_top_neighbour( 0 ), right_top_neighbour( 0 ),
      subdomain_x_size( 0 ), subdomain_y_size( 0 ), subdomain_z_size( 0 ), overlap_width( 0 ),
      left_overlap( 0 ), right_overlap( 0 ),
      bottom_overlap( 0 ), top_overlap( 0 ),
      closer_overlap( 0 ), further_overlap( 0 ),
      left_send_buff( 0 ), right_send_buff( 0 ),
      bottom_send_buff( 0 ), top_send_buff( 0 ),
      closer_send_buff( 0 ), further_send_buff( 0 ),
      bottom_left_send_buff( 0 ),
      bottom_right_send_buff( 0 ),
      top_left_send_buff( 0 ),
      top_right_send_buff( 0 ),
      left_recieve_buff( 0 ), right_recieve_buff( 0 ),
      bottom_recieve_buff( 0 ), top_recieve_buff( 0 ),
      closer_recieve_buff( 0 ), further_recieve_buff( 0 ),
      bottom_left_recieve_buff( 0 ),
      bottom_right_recieve_buff( 0 ),
      top_left_recieve_buff( 0 ),
      top_right_recieve_buff( 0 )
      {};
 
template< typename Real, typename Device, typename Index >
bool tnlMPIMesh< 3, Real, Device, Index > :: Init( const tnlGridOld< 3, Real, Device, Index >& u,
                                                   int& _mesh_x_size,
                                                   int& _mesh_y_size,
                                                   int& _mesh_z_size,
                                                   Index _overlap_width,
                                                   int root,
                                                   MPI_Comm comm )
   {
      dbgFunctionName( "tnlMPIMesh3D", "Init" );
#ifdef HAVE_MPI
      dbgMPIBarrier;
      dbgCout( "Getting MPI mesh dimenions..." );
      original_comm = comm;
      overlap_width = _overlap_width;
      mesh_x_size = _mesh_x_size;
      mesh_y_size = _mesh_y_size;
      mesh_z_size = _mesh_z_size;
      :: MPIBcast< int >( mesh_x_size, 1, 0 );
      :: MPIBcast< int >( mesh_y_size, 1, 0 );
      :: MPIBcast< int >( mesh_z_size, 1, 0 );
      int dims[ 3 ];
      dims[ 0 ] = mesh_x_size;
      dims[ 1 ] = mesh_y_size;
      dims[ 2 ] = mesh_y_size;
      if( ! mesh_x_size || ! mesh_y_size || ! mesh_z_size )
      {
         MPI_Dims_create( MPIGetSize( comm ), 3, dims );
         mesh_x_size = dims[ 0 ];
         mesh_y_size = dims[ 1 ];
         mesh_z_size = dims[ 2 ];
      }
      dbgMPIBarrier;
      dbgCout( "Mesh size is " << mesh_x_size << "x" << mesh_y_size << "x" << mesh_z_size );

      dbgMPIBarrier;
      dbgCout( "Creating cartesian MPI mesh ..." );
      int periods[ 3 ] = { false, false, false };
      MPI_Cart_create( comm, 3, dims, periods, true, &mesh_comm );
      int topo_type;
      MPI_Topo_test( mesh_comm, &topo_type );
      if( mesh_comm == MPI_COMM_NULL )
      {
         if( MPIGetRank( comm ) == root )
            std::cerr << "Not enough nodes for creating mesh " << mesh_x_size <<
                    "x" << mesh_y_size << "x" << mesh_z_size << std::endl;
         return false;
      }
 
      dbgMPIBarrier;
      dbgCout( "Getting node position in the MPI mesh ..." );
      MPI_Cart_coords( mesh_comm, MPIGetRank( mesh_comm ), 3, dims );
      node_x_pos = dims[ 0 ];
      node_y_pos = dims[ 1 ];
      node_z_pos = dims[ 2 ];
      dbgMPIBarrier;
      dbgCout( "Node position is ( " << node_x_pos << ", " << node_y_pos << ", " << node_z_pos << ")" );
 

      dbgMPIBarrier;
      dbgCout( "Checking MPI mesh neighbours ... " );
 
      MPI_Cart_shift( mesh_comm, 0, 1, &left_neighbour, &right_neighbour );
      MPI_Cart_shift( mesh_comm, 1, 1, &bottom_neighbour, &top_neighbour );
      MPI_Cart_shift( mesh_comm, 2, 1, &closer_neighbour, &further_neighbour );
      :: MPIBcast< int >( _overlap_width, 1, root );
      if( left_neighbour != MPI_PROC_NULL ) left_overlap = _overlap_width;
      else left_overlap = 0;
      if( right_neighbour != MPI_PROC_NULL ) right_overlap = _overlap_width;
      else right_overlap = 0;
      if( bottom_neighbour != MPI_PROC_NULL ) bottom_overlap = _overlap_width;
      else bottom_overlap = 0;
      if( top_neighbour != MPI_PROC_NULL ) top_overlap = _overlap_width;
      else top_overlap = 0;
      if( closer_neighbour != MPI_PROC_NULL ) closer_overlap = _overlap_width;
      else closer_overlap = 0;
      if( further_neighbour != MPI_PROC_NULL ) further_overlap = _overlap_width;
      else further_overlap = 0;

      dbgMPIBarrier;
      dbgCout( "Left " << left_overlap <<
               " Right " << right_overlap <<
               " Bottom " << bottom_overlap <<
               " Top " << top_overlap <<
               " Closer " << closer_overlap <<
               " Further " << further_overlap );

      /*int coords[ 3 ];
      if( node_x_pos < mesh_x_size - 1 && node_y_pos < mesh_y_size - 1 && node_z_pos < mesh_z_size - 1 )
      {
         coords[ 0 ] = node_x_pos + 1;
         coords[ 1 ] = node_y_pos + 1;
         coords[ 2 ] = node_z_pos + 1;
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
      else left_bottom_neighbour = MPI_PROC_NULL;*/
      // TODO: add for 3d
 
      dbgMPIBarrier;
      dbgCout( "Getting subdomain dimension ... " );
      domain_x_size = u. GetXSize();
      domain_y_size = u. GetYSize();
      domain_z_size = u. GetZSize();
      :: MPIBcast< int >( domain_x_size, 1, 0 );
      :: MPIBcast< int >( domain_y_size, 1, 0 );
      :: MPIBcast< int >( domain_z_size, 1, 0 );
 
      subdomain_x_size = domain_x_size / mesh_x_size;
      subdomain_y_size = domain_y_size / mesh_y_size;
      subdomain_z_size = domain_z_size / mesh_z_size;
 
      if( node_x_pos == mesh_x_size - 1 )
         subdomain_x_size = domain_x_size - subdomain_x_size * ( mesh_x_size - 1 );
 
      if( node_y_pos == mesh_y_size - 1 )
         subdomain_y_size = domain_y_size - subdomain_y_size * ( mesh_y_size - 1 );
 
      if( node_z_pos == mesh_z_size - 1 )
         subdomain_z_size = domain_z_size - subdomain_z_size * ( mesh_z_size - 1 );

      dbgMPIBarrier;
      dbgCout( "Subdomain dimensions are " << subdomain_x_size << "x"
               << subdomain_y_size << "x" << subdomain_z_size );
 
      Ax = u. GetAx();
      Ay = u. GetAy();
      Bx = u. GetBx();
      By = u. GetBy();
      Az = u. GetAz();
      Bz = u. GetBz();
 
      :: MPIBcast< double >( Ax, 1, 0 );
      :: MPIBcast< double >( Ay, 1, 0 );
      :: MPIBcast< double >( Az, 1, 0 );
      :: MPIBcast< double >( Bx, 1, 0 );
      :: MPIBcast< double >( By, 1, 0 );
      :: MPIBcast< double >( Bz, 1, 0 );
 
 
      FreeBuffers();
      if( left_overlap )
      {
         left_send_buff = new T[ subdomain_y_size * subdomain_z_size * left_overlap ];
         left_recieve_buff = new T[ subdomain_y_size * subdomain_z_size * left_overlap ];
      }
      if( right_overlap )
      {
         right_send_buff = new T[ subdomain_y_size * subdomain_z_size * right_overlap ];
         right_recieve_buff = new T[ subdomain_y_size * subdomain_z_size * right_overlap ];
      }
      if( bottom_overlap )
      {
         bottom_send_buff = new T[ subdomain_x_size * subdomain_z_size * bottom_overlap ];
         bottom_recieve_buff = new T[ subdomain_x_size * subdomain_z_size * bottom_overlap ];
      }
      if( top_overlap )
      {
         top_send_buff = new T[ subdomain_x_size * subdomain_z_size * top_overlap ];
         top_recieve_buff = new T[ subdomain_x_size * subdomain_z_size * top_overlap ];
      }
      if( closer_overlap )
      {
         closer_send_buff = new T[ subdomain_x_size * subdomain_y_size * closer_overlap ];
         closer_recieve_buff = new T[ subdomain_x_size * subdomain_y_size * closer_overlap ];
      }
      if( further_overlap )
      {
         further_send_buff = new T[ subdomain_x_size * subdomain_y_size * further_overlap ];
         further_recieve_buff = new T[ subdomain_x_size * subdomain_y_size * further_overlap ];
      }

      // TODO: fix it for 3D
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
     std::cout << "Node " << MPIGetRank()
           << " has position (" << GetXPos()
           << ", " << GetYPos()
           << ", " << GetZPos()
           << ") and dimensions " << GetSubdomainXSize()
           << " x " << GetSubdomainYSize()
           << " x " << GetSubdomainZSize() << std::endl;
#else
   domain_x_size = u. getDimensions(). x();
   domain_y_size = u. getDimensions(). y();
   domain_z_size = u. getDimensions(). z();
   Ax = u. getDomainLowerCorner(). x();
   Ay = u. getDomainLowerCorner(). y();
   Az = u. getDomainLowerCorner(). z();
   Bx = u. getDomainUpperCorner(). x();
   By = u. getDomainUpperCorner(). y();
   Bz = u. getDomainUpperCorner(). z();
#endif
      return true;
   };

template< typename Real, typename Device, typename Index >
bool tnlMPIMesh< 3, Real, Device, Index > :: Init( const tnlGridOld< 3, Real, Device, Index >& u,
                                                   const tnlParameterContainer& parameters,
                                                   Index _overlap_width,
                                                   int root,
                                                   MPI_Comm comm )
{
   int mpi_mesh_x_size = parameters. getParameter< int >( "mpi-mesh-x-size" );
   int mpi_mesh_y_size = parameters. getParameter< int >( "mpi-mesh-y-size" );
   int mpi_mesh_z_size = parameters. getParameter< int >( "mpi-mesh-z-size" );

   return Init( u,
                mpi_mesh_x_size,
                mpi_mesh_y_size,
                mpi_mesh_z_size,
                _overlap_width,
                root,
                comm );
}

 
template< typename Real, typename Device, typename Index >
bool tnlMPIMesh< 3, Real, Device, Index > :: CreateMesh( const tnlGridOld< 3, Real, Device, Index >& u,
                                                         tnlGridOld< 3, Real, Device, Index >& sub_u,
                                                         int root ) const
{
   dbgFunctionName( "tnlMPIMesh3D", "CreateMesh" );
#ifdef HAVE_MPI
   dbgMPIBarrier;
   dbgCout( "Creating subdomains ... " );
   double ax, ay, az, hx, hy, hz;
   tnlString name;
   int rank;
   if( MPIGetRank( original_comm ) == root )
   {
      assert( u );
      ax = u. GetAx();
      ay = u. GetAy();
      az = u. GetAz();
      hx = u. GetHx();
      hy = u. GetHy();
      hz = u. GetHz();
   }
   :: MPIBcast< double >( ax, 1, root, original_comm );
   :: MPIBcast< double >( ay, 1, root, original_comm );
   :: MPIBcast< double >( az, 1, root, original_comm );
   :: MPIBcast< double >( hx, 1, root, original_comm );
   :: MPIBcast< double >( hy, 1, root, original_comm );
   :: MPIBcast< double >( hz, 1, root, original_comm );
   name. MPIBcast( root, original_comm );
   dbgMPIBarrier;
   dbgCout( "Global domain is as: Ax = " << ax <<
                                " Ay = " << ay <<
                                " Az = " << az <<
                                " hx = " << hx <<
                                " hy = " << hy <<
                                " hz = " << hz );


   int err( 0 ), all_err( 0 );
   sub_u. SetNewDimensions( subdomain_x_size + left_overlap + right_overlap,
                            subdomain_y_size + bottom_overlap + top_overlap,
                            subdomain_z_size + closer_overlap + further_overlap );
   sub_u. SetNewDomain( ax + ( node_x_pos * subdomain_x_size - left_overlap ) * hx,
                        ax + ( ( node_x_pos + 1 ) * subdomain_x_size + right_overlap - 1 ) * hx,
                        ay + ( node_y_pos * subdomain_y_size - bottom_overlap ) * hy,
                        ay + ( ( node_y_pos + 1 ) * subdomain_y_size + top_overlap - 1 ) * hy,
                        az + ( node_z_pos * subdomain_z_size - closer_overlap ) * hz,
                        az + ( ( node_z_pos + 1 ) * subdomain_z_size + further_overlap - 1 ) * hz,
                        hx, hy, hz );
   if( ! sub_u )
   {
      std::cerr << "Unable to allocate subdomain grids for '" << name
           << "' on the node ( " << node_x_pos << ", " << node_y_pos
           << " rank " << MPIGetRank( original_comm ) << "." << std::endl;
      err = 1;
   }
   dbgMPIBarrier;
   dbgCout( "Subdomain is as: Ax = " << sub_u. GetAx() <<
                            " Ay = " << sub_u. GetAy() <<
                            " Az = " << sub_u. GetAz() <<
                            " hx = " << sub_u. GetHx() <<
                            " hy = " << sub_u. GetHy() <<
                            " hz = " << sub_u. GetHz() );

   MPI_Allreduce( &err, &all_err, 1, MPI_INT,MPI_SUM, mesh_comm );
   if( all_err != 0 ) return false;
#else
   sub_u. setLike( u );
#endif
return true;
};

template< typename Real, typename Device, typename Index >
void tnlMPIMesh< 3, Real, Device, Index > :: ScatterToNode( const tnlGridOld< 3, Real, Device, Index >& u,
                                                            tnlGridOld< 3, Real, Device, Index >& sub_u,
                                                            int dest_node,
                                                            int root ) const
{
   dbgFunctionName( "tnlMPIMesh3D", "ScatterToNode" );
#ifdef HAVE_MPI
   assert( sub_u );
   if( MPIGetRank( original_comm ) == root )
   {
      assert( u );
      dbgCout( "Node " << MPIGetRank() << " scatters to " << dest_node );
      int dest_x_pos;
      int dest_y_pos;
      int dest_z_pos;
      int coords[ 3 ];
      MPI_Cart_coords( mesh_comm, dest_node, 3, coords );
      dest_x_pos = coords[ 0 ];
      dest_y_pos = coords[ 1 ];
      dest_z_pos = coords[ 2 ];

      int dest_left_overlap( 0 ), dest_right_overlap( 0 ),
          dest_bottom_overlap( 0 ), dest_top_overlap( 0 ),
          dest_closer_overlap( 0 ), dest_further_overlap( 0 );
      dbgExpr( dest_node );
      if( dest_x_pos > 0 ) dest_left_overlap = overlap_width;
      if( dest_x_pos < mesh_x_size - 1 ) dest_right_overlap = overlap_width;
      if( dest_y_pos > 0 ) dest_bottom_overlap = overlap_width;
      if( dest_y_pos < mesh_y_size - 1 ) dest_top_overlap = overlap_width;
      if( dest_z_pos > 0 ) dest_closer_overlap = overlap_width;
      if( dest_z_pos < mesh_z_size - 1 ) dest_further_overlap = overlap_width;
 
      dbgCout( "Dest edges:  Lft. " << dest_left_overlap <<
                           " Rght. " << dest_right_overlap <<
                           " Btm. " << dest_bottom_overlap <<
                           " Top. " << dest_top_overlap <<
                           " Clsr. " << dest_closer_overlap <<
                           " Frth. " << dest_further_overlap );

      tnlGridOld< 3, Real, Device, Index >* mpi_buff;
      if( dest_node == root )
      {
         dbgCout( "Forwarding mpi_buffer to sub_u ..." );
         mpi_buff = &sub_u;
      }
      else
      {
          dbgCout( "Allocating MPI buffer - dimensions are: "
                   << subdomain_x_size + dest_left_overlap + dest_right_overlap << "x"
                   << subdomain_y_size + dest_bottom_overlap + dest_top_overlap << "x"
                   << subdomain_z_size + dest_closer_overlap + dest_further_overlap );
         mpi_buff = new tnlGridOld< 3, Real, Device, Index > ( subdomain_x_size + dest_left_overlap + dest_right_overlap,
                                       subdomain_y_size + dest_bottom_overlap + dest_top_overlap,
                                       subdomain_z_size + dest_closer_overlap + dest_further_overlap,
                                       0.0, 1.0, 0.0, 1.0, 0.0, 1.0 );
         dbgExpr( mpi_buff -> GetSize() );
         if( ! mpi_buff )
         {
            std::cerr << "Unable to allocate MPI buffer." << std::endl;
            abort();
         }
      }
      int i, j, k;
      const int i1 = dest_x_pos * subdomain_x_size - dest_left_overlap;
      const int i2 = ( dest_x_pos + 1 ) * subdomain_x_size + dest_right_overlap;
      const int j1 = dest_y_pos * subdomain_y_size - dest_bottom_overlap;
      const int j2 = ( dest_y_pos + 1 ) * subdomain_y_size + dest_top_overlap;
      const int k1 = dest_z_pos * subdomain_z_size - dest_closer_overlap;
      const int k2 = ( dest_z_pos + 1 ) * subdomain_z_size + dest_further_overlap;
      dbgCout( "Copying data to buffer ... " );
      dbgCout( "Limits are i @ [" << i1 << "," << i2 << "] " <<
                         " j @ [" << j1 << "," << j2 << "] " <<
                         " k @ [" << k1 << "," << k2 << "]" );
      for( i = i1; i < i2; i ++ )
         for( j = j1; j < j2; j ++ )
            for( k = k1; k < k2; k ++ )
            {
               //cout << i << " " << j << " " << k << " |";
               ( *mpi_buff )( i - i1, j - j1, k - k1 ) = u( i, j, k );
            }
      dbgCout( "Data succesfuly copied to buffer." );
      if( dest_node != root )
      {
         int buf_size = mpi_buff -> GetSize();
         dbgCout( "Calling MPI_Send and sending " << buf_size << "*" << sizeof( T ) << " bytes ... " );
         MPI_Send( mpi_buff -> getData(),
                   buf_size * sizeof( T ),
                   MPI_BYTE,
                   dest_node,
                   0,
                   mesh_comm );
         delete mpi_buff;
         dbgCout( "Data succesfuly sent. " );
      }
      return;
   }
   if( dest_node == root ) return;
   int buf_size = sub_u. GetSize();
   dbgCout( "Receiving data - " << buf_size << "*" << sizeof( T ) << " bytes required ..." );
   MPI_Status status;
   MPI_Recv( sub_u. getData(),
             buf_size * sizeof( T ),
             MPI_BYTE,
             0,
             0,
             mesh_comm,
             &status );
   dbgCout( "Data succesfuly received." );
#endif
}

template< typename Real, typename Device, typename Index >
void tnlMPIMesh< 3, Real, Device, Index > :: Scatter( const tnlGridOld< 3, Real, Device, Index >& u,
                                                      tnlGridOld< 3, Real, Device, Index >& sub_u,
                                                      int root ) const
{
   dbgFunctionName( "tnlMPIMesh3D", "Scatter" );
#ifdef HAVE_MPI
   dbgMPIBarrier;
   if( MPIGetRank( original_comm ) == root )
   {
      assert( u );
      int dest, mesh_size = mesh_x_size * mesh_y_size * mesh_z_size;
      for( dest = 0; dest < mesh_size; dest ++ )
      {
         dbgCout( "Scattering data to node " << dest << " ... " );
         ScatterToNode( u, sub_u, dest, root );
         dbgCout( "Scattering data to node " << dest << " done." );
      }
   }
   else
   {
       dbgCout( "Receiving data on node " << MPIGetRank( original_comm ) );
       ScatterToNode( u, sub_u, MPI_PROC_NULL, root );
   }
#else
   if( &u == &sub_u ) return;
   sub_u = u;
#endif
}
 
template< typename Real, typename Device, typename Index >
void tnlMPIMesh< 3, Real, Device, Index > :: Gather( tnlGridOld< 3, Real, Device, Index >& u,
                                                     const tnlGridOld< 3, Real, Device, Index >& sub_u,
                                                     int root ) const
{
   dbgFunctionName( "tnlMPIMesh3D", "Gather" );
#ifdef HAVE_MPI
   dbgCout( "Gathering data ..." );
   dbgMPIBarrier;
   if( MPIGetRank( original_comm ) == root )
   {
      int src, mesh_size = mesh_x_size * mesh_y_size * mesh_z_size;
      int i, j, k;
      MPI_Status status;
      for( src = 0; src < mesh_size; src ++ )
      {
         int coords[ 3 ];
         MPI_Cart_coords( mesh_comm, src, 3, coords );
         int src_x_pos = coords[ 0 ];
         int src_y_pos = coords[ 1 ];
         int src_z_pos = coords[ 2 ];

         dbgExpr( src );
         dbgExpr( src_x_pos );
         dbgExpr( src_y_pos );
         dbgExpr( src_z_pos );

         int src_left_overlap( 0 ), src_right_overlap( 0 ),
             src_bottom_overlap( 0 ), src_top_overlap( 0 ),
             src_closer_overlap( 0 ), src_further_overlap( 0 );
         if( src_x_pos > 0 ) src_left_overlap = overlap_width;
         if( src_x_pos < mesh_x_size - 1 ) src_right_overlap = overlap_width;
         if( src_y_pos > 0 ) src_bottom_overlap = overlap_width;
         if( src_y_pos < mesh_y_size - 1 ) src_top_overlap = overlap_width;
         if( src_z_pos > 0 ) src_closer_overlap = overlap_width;
         if( src_z_pos < mesh_z_size - 1 ) src_further_overlap = overlap_width;
         if( src != root )
         {
 
            dbgCout( "Allocating supporting buffer < " <<
                      src_x_pos * subdomain_x_size - src_left_overlap <<
                      ", " << ( src_x_pos + 1 ) * subdomain_x_size + src_right_overlap <<
                      " >x< " << src_y_pos * subdomain_y_size - src_bottom_overlap <<
                      ", " << ( src_y_pos + 1 ) * subdomain_y_size + src_top_overlap <<
                      " >" << src_z_pos * subdomain_z_size - src_closer_overlap <<
                      ", " << ( src_z_pos + 1 ) * subdomain_z_size + src_further_overlap << " >" );
 
            tnlGridOld< 3, Real, Device, Index > mpi_buff( subdomain_x_size + src_left_overlap + src_right_overlap,
                                   subdomain_y_size + src_bottom_overlap + src_top_overlap,
                                   subdomain_z_size + src_closer_overlap + src_further_overlap,
                                   0.0, 1.0, 0.0, 1.0, 0.0, 1.0 );
            int buf_size = mpi_buff. GetSize();
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
            const int k1 = src_z_pos * subdomain_z_size;
            const int k2 = k1 + subdomain_z_size;
            for( i = i1; i < i2; i ++ )
               for( j = j1; j < j2; j ++ )
                  for( k = k1; k < k2; k ++ )
                  {
                     //cout << "Node recv" << MPIGetRank( original_comm ) << " i = " << i << " j = " << j << std::endl;
                     u( i, j, k ) = mpi_buff( i - i1 + src_left_overlap,
                                              j - j1 + src_bottom_overlap,
                                              k - k1 + src_closer_overlap );
                  }
         }
         else
         {
            const int i1 = src_x_pos * subdomain_x_size;
            const int i2 = i1 + subdomain_x_size;
            const int j1 = src_y_pos * subdomain_y_size;
            const int j2 = j1 + subdomain_y_size;
            const int k1 = src_z_pos * subdomain_z_size;
            const int k2 = k1 + subdomain_z_size;
            for( i = i1; i < i2; i ++ )
               for( j = j1; j < j2; j ++ )
                  for( k = k1; k < k2; k ++ )
                  {
                     //cout << "Node cp" << MPIGetRank( original_comm ) << " i = " << i << " j = " << j << std::endl;
                     u( i, j, k ) = sub_u( i - i1 + src_left_overlap,
                                           j - j1 + bottom_overlap,
                                           k - k1 + closer_overlap );
                  }
         }
      }
   }
   else
   {
      dbgCout( "Sending data ... " );
      int buf_size = ( subdomain_x_size + left_overlap + right_overlap ) *
                          ( subdomain_y_size + bottom_overlap + top_overlap ) *
                          ( subdomain_z_size + closer_overlap + further_overlap );
      MPI_Send( const_cast< T* >( sub_u. getData() ),
                buf_size * sizeof( T ),
                MPI_BYTE,
                root,
                0,
                mesh_comm );
      dbgCout( "Sending succesfuly data done." );
   }
   dbgCout( "Gathering data done." );
#else
   if( &u == &sub_u ) return;
   u = sub_u;
#endif
}

template< typename Real, typename Device, typename Index >
void tnlMPIMesh< 3, Real, Device, Index > :: Synchronize( tnlGridOld< 3, Real, Device, Index >& u )
{
   dbgFunctionName( "tnlMPIMesh3D", "Synchronize" );
#ifdef HAVE_MPI
  int i, j, k;
  int min_x = left_overlap;
  int min_y = bottom_overlap;
  int min_z = closer_overlap;
  int max_x = min_x + subdomain_x_size;
  int max_y = min_y + subdomain_y_size;
  int max_z = min_z + subdomain_z_size;
  int wdth = overlap_width;
  MPI_Status status;
 
  MPI_Request lft_snd_rqst, rght_snd_rqst,
              lwr_snd_rqst, uppr_snd_rqst,
              clsr_snd_rqst, frth_snd_rqst,
              lwr_lft_snd_rqst, lwr_rght_snd_rqst,
              uppr_lft_snd_rqst, uppr_rght_snd_rqst,
              lft_rcv_rqst, rght_rcv_rqst,
              lwr_rcv_rqst, uppr_rcv_rqst,
              clsr_rcv_rqst, frth_rcv_rqst,
              lwr_lft_rcv_rqst, lwr_rght_rcv_rqst,
              uppr_lft_rcv_rqst, uppr_rght_rcv_rqst;
 
 
  dbgMPIBarrier;
 
  int buff_iter;
  dbgCout( "Starting communication with the left neighbour ... " );
  if( left_neighbour != MPI_PROC_NULL )
  {
     dbgCout( "Node ( " << node_x_pos << ", " << node_y_pos << ", " << node_z_pos << " ) - is SENDING data to the LEFT neighbour" );
     buff_iter = 0;
     for( i = 0; i < wdth; i ++ )
        for( j = 0; j < subdomain_y_size; j ++ )
           for( k = 0; k < subdomain_z_size; k ++ )
              left_send_buff[ buff_iter ++ ] = u( i + min_x, j + min_y, k + min_z );
     MPI_Isend( left_send_buff,
                wdth * subdomain_y_size * subdomain_z_size * sizeof( T ),
                MPI_BYTE,
                left_neighbour,
                0,
                mesh_comm ,
                &lft_snd_rqst );
 
     dbgCout( "Node ( " << node_x_pos << ", " << node_y_pos << ", " << node_z_pos << " ) - is RECEIVING data from the LEFT neighbour" );
     MPI_Irecv( left_recieve_buff,
                wdth * subdomain_y_size * subdomain_z_size * sizeof( T ),
                MPI_CHAR,
                left_neighbour,
                0,
                mesh_comm,
                &lft_rcv_rqst );
  }
  dbgMPIBarrier;
 
  dbgCout( "Starting communication with the right neighbour ... " );
  if( right_neighbour != MPI_PROC_NULL )
  {
     dbgCout( "Node ( " << node_x_pos << ", " << node_y_pos << ", " << node_z_pos << " ) - is SENDING data to the RIGHT neighbour" );
     buff_iter = 0;
     for( i = 0; i < wdth; i ++ )
        for( j = 0; j < subdomain_y_size; j ++ )
           for( k = 0; k < subdomain_z_size; k ++ )
              right_send_buff[ buff_iter ++ ] = u( max_x - wdth + i, j + min_y, k + min_z );
     MPI_Isend( right_send_buff,
                wdth * subdomain_y_size * subdomain_z_size * sizeof( T ),
                MPI_BYTE,
                right_neighbour,
                0,
                mesh_comm,
                &rght_snd_rqst );
 
     dbgCout( "Node ( " << node_x_pos << ", " << node_y_pos << ", " << node_z_pos << " ) - is RECEIVING data from the RIGHT neighbour" );
     MPI_Irecv( right_recieve_buff,
                wdth * subdomain_y_size * subdomain_z_size * sizeof( T ),
                MPI_BYTE,
                right_neighbour,
                0,
                mesh_comm,
                &rght_rcv_rqst );
  }
  dbgMPIBarrier;
 
  dbgCout( "Starting communication with the bottom neighbour ... " );
  if( bottom_neighbour != MPI_PROC_NULL )
  {
     dbgCout( "Node ( " << node_x_pos << ", " << node_y_pos << ", " << node_z_pos << " ) - is SENDING data to the BOTTOM neighbour" );
     buff_iter = 0;
     for( i = 0; i < subdomain_x_size; i ++ )
        for( j = 0; j < wdth; j ++ )
           for( k = 0; k < subdomain_z_size; k ++ )
              bottom_send_buff[ buff_iter ++ ] = u( min_x + i, min_y + j, min_z + k );
     MPI_Isend( bottom_send_buff,
                wdth * subdomain_x_size * subdomain_z_size * sizeof( T ),
                MPI_BYTE,
                bottom_neighbour,
                0,
                mesh_comm,
                &lwr_snd_rqst );
 
     dbgCout( "Node ( " << node_x_pos << ", " << node_y_pos << ", " << node_z_pos << " ) - is RECEIVING data from the BOTTOM neighbour" );
     MPI_Irecv( bottom_recieve_buff,
                wdth * subdomain_y_size * subdomain_z_size * sizeof( T ),
                MPI_BYTE,
                bottom_neighbour,
                0,
                mesh_comm,
                &lwr_rcv_rqst );
  }
  dbgMPIBarrier;

  dbgCout( "Starting communication with the uppper neighbour ... " );
  if( top_neighbour != MPI_PROC_NULL )
  {
     dbgCout( "Node ( " << node_x_pos << ", " << node_y_pos << ", " << node_z_pos << " ) - is SENDING data to the TOP neighbour" );
     buff_iter = 0;
     for( i = 0; i < subdomain_x_size; i ++ )
        for( j = 0; j < wdth; j ++ )
           for( k = 0; k < subdomain_z_size; k ++ )
              top_send_buff[ buff_iter ++ ] = u( min_x + i, max_y - wdth + j, min_z + k );
     MPI_Isend( top_send_buff,
                wdth * subdomain_x_size * subdomain_z_size * sizeof( T ),
                MPI_BYTE,
                top_neighbour,
                0,
                mesh_comm,
                &uppr_snd_rqst );
 
     dbgCout( "Node ( " << node_x_pos << ", " << node_y_pos << ", " << node_z_pos << " ) - RECEIVING data from the TOP neighbour" );
     MPI_Irecv( top_recieve_buff,
                wdth * subdomain_y_size * subdomain_z_size * sizeof( T ),
                MPI_BYTE,
                top_neighbour,
                0,
                mesh_comm,
                &uppr_rcv_rqst );
  }
  dbgMPIBarrier;
 
  dbgCout( "Starting communication with the closer neighbour ... " );
  if( closer_neighbour != MPI_PROC_NULL )
  {
     dbgCout( "Node ( " << node_x_pos << ", " << node_y_pos << ", " << node_z_pos << " ) - is SENDING data to the CLOSER neighbour" );
     buff_iter = 0;
     for( i = 0; i < subdomain_x_size; i ++ )
        for( j = 0; j < subdomain_y_size; j ++ )
           for( k = 0; k < wdth; k ++ )
              closer_send_buff[ buff_iter ++ ] = u( min_x + i, min_y + j, min_z + k );
     MPI_Isend( closer_send_buff,
                wdth * subdomain_x_size * subdomain_y_size * sizeof( T ),
                MPI_BYTE,
                closer_neighbour,
                0,
                mesh_comm,
                &clsr_snd_rqst );
 
     dbgCout( "Node ( " << node_x_pos << ", " << node_y_pos << ", " << node_z_pos << " ) - is RECEIVING data from the CLOSER neighbour" );
     MPI_Irecv( closer_recieve_buff,
                wdth * subdomain_x_size * subdomain_y_size * sizeof( T ),
                MPI_BYTE,
                closer_neighbour,
                0,
                mesh_comm,
                &clsr_rcv_rqst );
  }
  dbgMPIBarrier;

  dbgCout( "Starting communication with the further neighbour ... " );
  if( further_neighbour != MPI_PROC_NULL )
  {
     dbgCout( "Node ( " << node_x_pos << ", " << node_y_pos << ", " << node_z_pos << " ) - is SENDING data to the FURTHER neighbour" );
     buff_iter = 0;
     for( i = 0; i < subdomain_x_size; i ++ )
        for( j = 0; j < subdomain_y_size; j ++ )
           for( k = 0; k < wdth; k ++ )
              further_send_buff[ buff_iter ++ ] = u( min_x + i, min_y + j, max_z - wdth + k );
     MPI_Isend( further_send_buff,
                wdth * subdomain_x_size * subdomain_y_size * sizeof( T ),
                MPI_BYTE,
                further_neighbour,
                0,
                mesh_comm,
                &frth_snd_rqst );
 
     dbgCout( "Node ( " << node_x_pos << ", " << node_y_pos << ", " << node_z_pos << " ) - is RECEIVING data from the FURTHER neighbour" );
     MPI_Irecv( further_recieve_buff,
                wdth * subdomain_x_size * subdomain_y_size * sizeof( T ),
                MPI_BYTE,
                further_neighbour,
                0,
                mesh_comm,
                &frth_rcv_rqst );
  }
  dbgMPIBarrier;
 
  int wdth_2 = wdth * wdth;
  /*
  // starting communication with lower left neighbour
  if( left_bottom_neighbour != MPI_PROC_NULL )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - SENDING small square to the BOTTOM LEFT neighbour." );
     //for( i = 0; i < wdth; i ++ )
     //   for( j = 0; j < wdth; j ++ )
     //     bottom_left_send_buff[ j * wdth + i ] =
     //         u( min_x + i, min_y + j );
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
  DBG_MPI_BARRIER;
 
  // starting communication with lower right neighbour
  if( right_bottom_neighbour != MPI_PROC_NULL )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - SENDING small square to the BOTTOM RIGHT neighbour." );
     //for( i = 0; i < wdth; i ++ )
     //   for( j = 0; j < wdth; j ++ )
     //      bottom_right_send_buff[ j * wdth + i ] =
     //         u( max_x - wdth + i, min_y + j );
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
  DBG_MPI_BARRIER;

  // starting communication with upper left neighbour
  if( left_top_neighbour != MPI_PROC_NULL )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - SENDING small square to the TOP LEFT neighbour." );
     //for( i = 0; i < wdth; i ++ )
     //   for( j = 0; j < wdth; j ++ )
     //      top_left_send_buff[ j * wdth + i ] =
     //         u( min_x + i, max_y - wdth + j );
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
  DBG_MPI_BARRIER;

  // starting communication with upper right neighbour
  if( right_top_neighbour != MPI_PROC_NULL )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - SENDING small square to the TOP RIGHT neighbour." );
     //for( i = 0; i < wdth; i ++ )
     //   for( j = 0; j < wdth; j ++ )
     //      top_right_send_buff[ j * wdth + i ] =
     //         u( max_x - wdth + i, max_y - wdth + j );
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
  DBG_MPI_BARRIER;
  */

  dbgCout( "Finishing communication with the left neighbour ... " );
  if( left_neighbour != MPI_PROC_NULL )
  {
     dbgCout( "Node ( " << node_x_pos << ", " << node_y_pos << ", " << node_z_pos << " ) - is WAITING for data from LEFT neighbour." );
     MPI_Wait( &lft_rcv_rqst, &status );
     buff_iter = 0;
     for( i = 0; i < wdth; i ++ )
        for( j = 0; j < subdomain_y_size; j ++ )
           for( k = 0; k < subdomain_z_size; k ++ )
              u( min_x - wdth + i, min_y + j, min_z + k ) = left_recieve_buff[ buff_iter ++ ];
     MPI_Wait( &lft_snd_rqst, &status );
  }
  dbgMPIBarrier;
 
  dbgCout( "Finishing communication with the right neighbour ..." );
  if( right_neighbour != MPI_PROC_NULL )
  {
     dbgCout( "Node ( " << node_x_pos << ", " << node_y_pos << ", " << node_z_pos << " ) - is WAITING for data from RIGHT neighbour." );
     MPI_Wait( &rght_rcv_rqst, &status );
     buff_iter = 0;
     for( i = 0; i < wdth; i ++ )
        for( j = 0; j < subdomain_y_size; j ++ )
           for( k = 0; k < subdomain_z_size; k ++ )
              u( max_x + i, min_y + j, min_z + k ) = right_recieve_buff[ buff_iter ++ ];
     MPI_Wait( &rght_snd_rqst, &status );
  }
  dbgMPIBarrier;
 
  dbgCout( "Finishing communication with the lower neighbour ... " );
  if( bottom_neighbour != MPI_PROC_NULL )
  {
     dbgCout( "Node ( " << node_x_pos << ", " << node_y_pos << ", " << node_z_pos << " ) - is WAITING for data from BOTTOM neighbour." );
     MPI_Wait( &lwr_rcv_rqst, &status );
     buff_iter = 0;
     for( i = 0; i < subdomain_x_size; i ++ )
        for( j = 0; j < wdth; j ++ )
           for( k = 0; k < subdomain_z_size; k ++ )
              u( min_x + i, min_y - wdth + j, min_z + k ) = bottom_recieve_buff[ buff_iter ++ ];
     MPI_Wait( &lwr_snd_rqst, &status );
  }
  dbgMPIBarrier;
 
  dbgCout( "Finishing communication with the upper neighbour ..." );
  if( top_neighbour != MPI_PROC_NULL )
  {
     dbgCout( "Node ( " << node_x_pos << ", " << node_y_pos << ", " << node_z_pos << " ) - is WAITING for data from TOP neighbour." );
     buff_iter = 0;
     MPI_Wait( &uppr_rcv_rqst, &status );
     for( i = 0; i < subdomain_x_size; i ++ )
        for( j = 0; j < wdth; j ++ )
           for( k = 0; k < subdomain_z_size; k ++ )
              u( min_x + i, max_y + j, min_z + k ) = top_recieve_buff[ buff_iter ++ ];
     MPI_Wait( &uppr_snd_rqst, &status );
  }
  dbgMPIBarrier;
 
  dbgCout( "Finishing communication with the closer neighbour ... " );
  if( closer_neighbour != MPI_PROC_NULL )
  {
     dbgCout( "Node ( " << node_x_pos << ", " << node_y_pos << ", " << node_z_pos << " ) - is WAITING for data from BOTTOM neighbour." );
     MPI_Wait( &clsr_rcv_rqst, &status );
     buff_iter = 0;
     for( i = 0; i < subdomain_x_size; i ++ )
        for( j = 0; j < subdomain_y_size; j ++ )
           for( k = 0; k < wdth; k ++ )
              u( min_x + i, min_y + j, min_z -wdth + k ) = closer_recieve_buff[ buff_iter ++ ];
     MPI_Wait( &clsr_snd_rqst, &status );
  }
  dbgMPIBarrier;
 
  dbgCout( "Finishing communication with the further neighbour ... " );
  if( further_neighbour != MPI_PROC_NULL )
  {
     dbgCout( "Node ( " << node_x_pos << ", " << node_y_pos << ", " << node_z_pos << " ) - is WAITING for data from FURTHER neighbour." );
     buff_iter = 0;
     MPI_Wait( &frth_rcv_rqst, &status );
     for( i = 0; i < subdomain_x_size; i ++ )
        for( j = 0; j < subdomain_y_size; j ++ )
           for( k = 0; k < wdth; k ++ )
              u( min_x + i, min_y + j, max_z + k ) = further_recieve_buff[ buff_iter ++ ];
     MPI_Wait( &frth_snd_rqst, &status );
  }
  dbgMPIBarrier;
 
 
  /*
  // finishing communication with the lower left neighbour
  if( left_bottom_neighbour != MPI_PROC_NULL  )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - WAITING for data from BOTTOM LEFT neighbour." );
     MPI_Wait( &lwr_lft_rcv_rqst, &status );
     //for( i = 0; i < wdth; i ++ )
     //   for( j = 0; j < wdth; j ++ )
     //      u( min_x - wdth + i, min_y - wdth + j ) =
     //      bottom_left_recieve_buff[ j * wdth + i ];
     MPI_Wait( &lwr_lft_snd_rqst, &status );
  }
  DBG_MPI_BARRIER;
 
  // finishing communication with the lower right neighbour
  if( right_bottom_neighbour != MPI_PROC_NULL )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - WAITING for data from BOTTOM RIGHT neighbour." );
     MPI_Wait( &lwr_rght_rcv_rqst, &status );
     //for( i = 0; i < wdth; i ++ )
     //   for( j = 0; j < wdth; j ++ )
     //      u( max_x + i, min_y - wdth + j ) =
     //      bottom_right_recieve_buff[ j * wdth + i ];
     MPI_Wait( &lwr_rght_snd_rqst, &status );
  }
  DBG_MPI_BARRIER;

  // finishing communication with the upper right neighbour
  if( right_top_neighbour != MPI_PROC_NULL )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - WAITING for data from TOP RIGHT neighbour." );
     MPI_Wait( &uppr_rght_rcv_rqst, &status );
     //for( i = 0; i < wdth; i ++ )
     //   for( j = 0; j < wdth; j ++ )
     //      u( max_x + i, max_y + j ) =
     //      top_right_recieve_buff[ j * wdth + i ];
     MPI_Wait( &uppr_rght_snd_rqst, &status );
  }
  DBG_MPI_BARRIER;
 
  // finishing communication with the upper left neighbour
  if( left_top_neighbour != MPI_PROC_NULL )
  {
     dbgCout( "node ( " << node_x_pos << ", " << node_y_pos << " ) - WAITING for data from TOP LEFT neighbour." );
     MPI_Wait( &uppr_lft_rcv_rqst, &status );
     //for( i = 0; i < wdth; i ++ )
     //   for( j = 0; j < wdth; j ++ )
     //      u( min_x - wdth + i, max_y + j ) =
     //      top_left_recieve_buff[ j * wdth + i ];
     MPI_Wait( &uppr_lft_snd_rqst, &status );
  }*/
 
  dbgCout( "Synchronisation done..." );
  dbgMPIBarrier;
#endif
}

template< typename Real, typename Device, typename Index >
void tnlMPIMesh< 3, Real, Device, Index > ::  FreeBuffers()
{
   if( left_send_buff ) delete left_send_buff;
   if( right_send_buff ) delete right_send_buff;
   if( bottom_send_buff ) delete bottom_send_buff;
   if( top_send_buff ) delete top_send_buff;
   if( closer_send_buff ) delete closer_send_buff;
   if( bottom_left_send_buff ) delete bottom_left_send_buff;
   if( bottom_right_send_buff ) delete bottom_right_send_buff;
   if( top_left_send_buff ) delete top_left_send_buff;
   if( top_right_send_buff ) delete top_right_send_buff;
   if( left_recieve_buff ) delete left_recieve_buff;
   if( right_recieve_buff ) delete right_recieve_buff;
   if( bottom_recieve_buff ) delete bottom_recieve_buff;
   if( top_recieve_buff ) delete top_recieve_buff;
   if( closer_recieve_buff ) delete closer_recieve_buff;
   if( further_recieve_buff ) delete further_recieve_buff;
   if( bottom_left_recieve_buff ) delete bottom_left_recieve_buff;
   if( bottom_right_recieve_buff ) delete bottom_right_recieve_buff;
   if( top_left_recieve_buff ) delete top_left_recieve_buff;
   if( top_right_recieve_buff ) delete top_right_recieve_buff;
   left_send_buff = 0;
   right_send_buff = 0;
   bottom_send_buff = 0;
   top_send_buff = 0;
   closer_send_buff = 0;
   further_send_buff = 0;
   bottom_left_send_buff = 0;
   bottom_right_send_buff = 0;
   top_left_send_buff = 0;
   top_right_send_buff = 0;
   left_recieve_buff = 0;
   right_recieve_buff = 0;
   bottom_recieve_buff = 0;
   top_recieve_buff = 0;
   closer_recieve_buff = 0;
   further_recieve_buff = 0;
   bottom_left_recieve_buff = 0;
   bottom_right_recieve_buff = 0;
   top_left_recieve_buff = 0;
   top_right_recieve_buff = 0;
};

template< typename Real, typename Device, typename Index >
void DrawSubdomains( const tnlMPIMesh< 3, Real, Device, Index >& mpi_mesh,
                     const tnlGridOld< 3, Real, Device, Index >& u,
                     const char* file_name_base,
                     const char* format )
{
   int num = mpi_mesh. GetXPos() * 100 + mpi_mesh. GetYPos() * 10 + mpi_mesh. GetZPos();
   tnlString file_name;
   FileNameBaseNumberEnding( file_name_base,
                             num,
                             3,
                             ".vti",
                             file_name );
   Draw( u, file_name. getString(), format );
};

#endif
