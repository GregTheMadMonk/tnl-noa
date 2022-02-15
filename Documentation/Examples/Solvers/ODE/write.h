template< typename Vector,
          typename Real = typename Vector::RealType,
          typename Index = typename Vector::IndexType >
void write( const Vector& u, const char* file_name, const Index n, const Real h, const Index output_idx )
{
   TNL::FileName output_file_name( file_name );
   output_file_name.setIndex( output_idx );
   output_file_name.setExtension( "txt" );
   std::cout << "Writing to file " << output_file_name.getFileName() << std::endl;
   std::fstream file;
   file.open( output_file_name.getFileName().data(), std::ios::out );
   for( Index i = 0; i < n; i++ )
      file << i*h << " " << u.getElement( i ) << std::endl;
}