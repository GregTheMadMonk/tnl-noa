/*
* Experimentální test pro implementaci MPI do TNL
 */

#include <iostream>
#include <mpi.h>

unsigned int errors=0;
unsigned int success=0;
#define TEST_TEST(a) if((a)){cout << __LINE__ <<":\t OK " <<endl;success++;}else{cout << __LINE__<<":\t FAIL" <<endl;errors++;}
#define TEST_RESULT cout<<"SUCCES: "<<success<<endl<<"ERRRORS: "<<errors<<endl;
inline void Test_Say( const char * message)
{
#ifdef TEST_VERBOSE
	cout << message <<endl;
#endif
}

using namespace std;

int main ( int argc, char *argv[])
{
  int rank, size;

  MPI_Init (&argc, &argv);      /* starts MPI */
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);        /* get current process id */
  MPI_Comm_size (MPI_COMM_WORLD, &size);        /* get number of processes */
  cout << "Hello world from process " << rank <<" of " << size << endl;
  MPI_Finalize();
  return 0;
}

