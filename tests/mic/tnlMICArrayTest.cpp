/**
 *  Experimentalní test pro získání zkušeností s TNL a (a strarým dobrým MIC, intel offloadingem a podobně) 
 */
 
#include <iostream>
#include <TNL/Devices/MIC.h>
#include <omp.h>
#include <TNL/Containers/Array.h>

	using namespace std;
	using namespace TNL;
	using namespace TNL::Containers;

//TUNE MACROS FOR YOUR FUNKY OUTPUT
#define TEST_VERBOSE

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


int main(void)
{
    cout << "Array on MIC test by hanouvit:" <<endl;
	
	#ifdef HAVE_ICPC
		cout << "ICPC in USE" <<endl;
	#endif

	#ifdef HAVE_MIC
		cout << "MIC in USE" <<endl; //LOL
	#endif
	
#ifdef HAVE_MIC
//prepare arrays with data
	
	Array<double,Devices::MIC,int> aa(10);
	Array<double,Devices::MIC,int> ee(6);
	Array<double,Devices::Host,int> cc(5);

//fill it 
Devices::MICHider<double> data_ptr;
data_ptr.pointer= aa.getData();	
int size=aa.getSize();
        
#pragma offload target(mic) in(data_ptr,size)
{
    for(int i=0;i<size;i++)
    {
            data_ptr.pointer[i]=i;
    }
}

for(int i=0;i<5;i++)
{
	cc[i]=10+i;
}

//prepare arrays for funky tests
Array<double,Devices::MIC,int> bb(10);
Array<double,Devices::MIC,int> dd(0);

//TEST IT!
Test_Say("aa.getSize():");
TEST_TEST(aa.getSize()==10);

Test_Say("Is aa filled correctly? (aa.getElement):");
for(int i=0;i<10;i++)
	TEST_TEST(aa.getElement(i)==i);

Test_Say("Copy to bb(MIC->MIC) (=):");
bb=aa;
TEST_TEST(aa.getSize()==bb.getSize());
for(int i=0;i<bb.getSize();i++)
	TEST_TEST(bb.getElement(i)==i);

Test_Say("setLike:");
bb.setLike(cc);
TEST_TEST(bb.getSize()==cc.getSize());
Test_Say("Copy (Host -> MIC) (=)");
bb=cc;
for(int i=0;i<bb.getSize();i++)
	TEST_TEST(bb.getElement(i)==i+10);

Test_Say("setValue:");
bb.setValue(5);
for(int i=0;i<bb.getSize();i++)
	TEST_TEST(bb.getElement(i)==5);

Test_Say("swap:");
aa.swap(bb);
TEST_TEST(aa.getSize()==5||bb.getSize()==10);
for(int i=0;i<aa.getSize();i++)
{
	TEST_TEST(aa.getElement(i)==5);
}
for(int i=0;i<bb.getSize();i++)
{
	TEST_TEST(bb.getElement(i)==i);
}

Test_Say("(MIC -> MIC) ==");
aa.setLike(bb);
aa=bb;
TEST_TEST(aa==bb);
TEST_TEST(!(aa!=bb));
TEST_TEST(aa!=ee);
bb.setElement(5,66);
TEST_TEST(aa!=bb);

Test_Say("(Host -> MIC) !=");
aa.setLike(cc);
aa=cc;
TEST_TEST(aa==cc);
aa.setElement(3,66);
TEST_TEST(aa!=cc);

Test_Say("bidn (light test)");
dd.bind(bb,5);
TEST_TEST(dd.getSize()==5);
TEST_TEST(dd.getElement(1)==6);

//Mylsím, že není zdaleka testováno vše...

///////////////////////////////////////////////////////////////////////////////

Test_Say("File Array Test: \n");

//prepare arrays with data

aa.setSize(10);
ee.setSize(6);
cc.setSize(5);

//fill it UP
/*Devices::MICHider<double> data_ptr;*/
data_ptr.pointer= aa.getData();	
//size=aa.getSize();

#pragma offload target(mic) in(data_ptr,size)
{
    for(int i=0;i<size;i++)
    {
            data_ptr.pointer[i]=i;
    }
}

for(int i=0;i<5;i++)
{
	cc[i]=10+i;
}

File soubor;
soubor.open("/tmp/tnlArrayExperimentSave_cc.bin",tnlWriteMode);
cc.save(soubor);
soubor.close();

soubor.open("/tmp/tnlArrayExperimentSave_aa.bin",tnlWriteMode);
aa.save(soubor);
soubor.close();

ee.bind(aa,5,5);
ee.boundLoad("/tmp/tnlArrayExperimentSave_cc.bin");

TEST_TEST( 10 == aa.getSize())
for(int i=0;i<5;i++)
{
	TEST_TEST(aa.getElement(i)==i)
}
for(int i=5;i<10;i++)
{
	TEST_TEST(aa.getElement(i)==i+5)
}

soubor.open("/tmp/tnlArrayExperimentSave_aa.bin",tnlReadMode);
cc.load(soubor);
soubor.close();

TEST_TEST( 10 == cc.getSize())
for(int i=0;i<cc.getSize();i++)
{
	TEST_TEST(cc.getElement(i)==i)
}

#endif 

	TEST_RESULT;
    return 0;
}
