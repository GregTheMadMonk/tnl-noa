/***************************************************************************
                          tnlParallelEikonalSolver2D_impl.h  -  description
                             -------------------
    begin                : Nov 28 , 2014
    copyright            : (C) 2014 by Tomas Sobotik
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef TNLPARALLELEIKONALSOLVER3D_IMPL_H_
#define TNLPARALLELEIKONALSOLVER3D_IMPL_H_


#include "tnlParallelEikonalSolver.h"
#include <core/mfilename.h>

template< typename SchemeHost, typename SchemeDevice, typename Device>
tnlParallelEikonalSolver<3,SchemeHost, SchemeDevice, Device, double, int>::tnlParallelEikonalSolver()
{
	cout << "a" << endl;
	this->device = tnlHostDevice;  /////////////// tnlCuda Device --- vypocet na GPU, tnlHostDevice   ---    vypocet na CPU

#ifdef HAVE_CUDA
	if(this->device == tnlCudaDevice)
	{
	run_host = 1;
	}
#endif

	cout << "b" << endl;
}

template< typename SchemeHost, typename SchemeDevice, typename Device>
void tnlParallelEikonalSolver<3,SchemeHost, SchemeDevice, Device, double, int>::test()
{
/*
	for(int i =0; i < this->subgridValues.getSize(); i++ )
	{
		insertSubgrid(getSubgrid(i), i);
	}
*/
}

template< typename SchemeHost, typename SchemeDevice, typename Device>

bool tnlParallelEikonalSolver<3,SchemeHost, SchemeDevice, Device, double, int>::init( const tnlParameterContainer& parameters )
{
	cout << "Initializating solver..." << endl;
	const tnlString& meshLocation = parameters.getParameter <tnlString>("mesh");
	this->mesh.load( meshLocation );

	this->n = parameters.getParameter <int>("subgrid-size");
	cout << "Setting N to " << this->n << endl;

	this->subMesh.setDimensions( this->n, this->n, this->n );
	this->subMesh.setDomain( tnlStaticVector<3,double>(0.0, 0.0, 0.0),
							 tnlStaticVector<3,double>(mesh.template getSpaceStepsProducts< 1, 0, 0 >()*(double)(this->n), mesh.template getSpaceStepsProducts< 0, 1, 0 >()*(double)(this->n),mesh.template getSpaceStepsProducts< 0, 0, 1 >()*(double)(this->n)) );

	this->subMesh.save("submesh.tnl");

	const tnlString& initialCondition = parameters.getParameter <tnlString>("initial-condition");
	this->u0.load( initialCondition );

	//cout << this->mesh.getCellCenter(0) << endl;

	this->delta = parameters.getParameter <double>("delta");
	this->delta *= mesh.template getSpaceStepsProducts< 1, 0, 0 >()*mesh.template getSpaceStepsProducts< 0, 1, 0 >();

	cout << "Setting delta to " << this->delta << endl;

	this->tau0 = parameters.getParameter <double>("initial-tau");
	cout << "Setting initial tau to " << this->tau0 << endl;
	this->stopTime = parameters.getParameter <double>("stop-time");

	this->cflCondition = parameters.getParameter <double>("cfl-condition");
	this -> cflCondition *= sqrt(mesh.template getSpaceStepsProducts< 1, 0, 0 >()*mesh.template getSpaceStepsProducts< 0, 1, 0 >());
	cout << "Setting CFL to " << this->cflCondition << endl;

	stretchGrid();
	this->stopTime /= (double)(this->gridCols);
	this->stopTime *= (1.0+1.0/((double)(this->n) - 2.0));
	cout << "Setting stopping time to " << this->stopTime << endl;
	//this->stopTime = 1.5*((double)(this->n))*parameters.getParameter <double>("stop-time")*mesh.template getSpaceStepsProducts< 1, 0, 0 >();
	//cout << "Setting stopping time to " << this->stopTime << endl;

	cout << "Initializating scheme..." << endl;
	if(!this->schemeHost.init(parameters))
	{
		cerr << "SchemeHost failed to initialize." << endl;
		return false;
	}
	cout << "Scheme initialized." << endl;

	test();

	VectorType* tmp = new VectorType[subgridValues.getSize()];
	bool containsCurve = false;

#ifdef HAVE_CUDA

	if(this->device == tnlCudaDevice)
	{
	/*cout << "Testing... " << endl;
	if(this->device == tnlCudaDevice)
	{
	if( !initCUDA3D(parameters, gridRows, gridCols) )
		return false;
	}*/
		//cout << "s" << endl;
	cudaMalloc(&(this->cudaSolver), sizeof(tnlParallelEikonalSolver<3,SchemeHost, SchemeDevice, Device, double, int >));
	//cout << "s" << endl;
	cudaMemcpy(this->cudaSolver, this,sizeof(tnlParallelEikonalSolver<3,SchemeHost, SchemeDevice, Device, double, int >), cudaMemcpyHostToDevice);
	//cout << "s" << endl;
	double** tmpdev = NULL;
	cudaMalloc(&tmpdev, sizeof(double*));
	//double* tmpw;
	cudaMalloc(&(this->tmpw), this->work_u.getSize()*sizeof(double));
	cudaMalloc(&(this->runcuda), sizeof(int));
	cudaDeviceSynchronize();
	checkCudaDevice;
	int* tmpUC;
	cudaMalloc(&(tmpUC), this->work_u.getSize()*sizeof(int));
	cudaMemcpy(tmpUC, this->unusedCell.getData(), this->unusedCell.getSize()*sizeof(int), cudaMemcpyHostToDevice);

	initCUDA3D<SchemeTypeHost, SchemeTypeDevice, DeviceType><<<1,1>>>(this->cudaSolver, (this->tmpw), (this->runcuda),tmpUC);
	cudaDeviceSynchronize();
	checkCudaDevice;
	//cout << "s " << endl;
	//cudaMalloc(&(cudaSolver->work_u_cuda), this->work_u.getSize()*sizeof(double));
	double* tmpu = NULL;

	cudaMemcpy(&tmpu, tmpdev,sizeof(double*), cudaMemcpyDeviceToHost);
	//printf("%p %p \n",tmpu,tmpw);
	cudaMemcpy((this->tmpw), this->work_u.getData(), this->work_u.getSize()*sizeof(double), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	checkCudaDevice;
	//cout << "s "<< endl;

	}
#endif

	if(this->device == tnlHostDevice)
	{
		for(int i = 0; i < this->subgridValues.getSize(); i++)
		{
//			cout << "Working on subgrid " << i <<" --- check 1" << endl;

			if(! tmp[i].setSize(this->n*this->n*this->n))
				cout << "Could not allocate tmp["<< i <<"] array." << endl;
//			cout << "Working on subgrid " << i <<" --- check 2" << endl;

			tmp[i] = getSubgrid(i);
			containsCurve = false;
//			cout << "Working on subgrid " << i <<" --- check 3" << endl;


			for(int j = 0; j < tmp[i].getSize(); j++)
			{
				if(tmp[i][0]*tmp[i][j] <= 0.0)
				{
					containsCurve = true;
					j=tmp[i].getSize();
//					cout << tmp[i][0] << " " << tmp[i][j] << endl;
				}

			}
//			cout << "Working on subgrid " << i <<" --- check 4" << endl;

			if(containsCurve)
			{
//				cout << "Computing initial SDF on subgrid " << i << "." << endl;
				tmp[i] = runSubgrid(0, tmp[i] ,i);
				insertSubgrid( tmp[i], i);
				setSubgridValue(i, 4);
//				cout << "Computed initial SDF on subgrid " << i  << "." << endl;
			}
			containsCurve = false;

		}
//		cout << "CPU: Curve found" << endl;
	}
#ifdef HAVE_CUDA
	else if(this->device == tnlCudaDevice)
	{
//		cout << "pre 1 kernel" << endl;
		cudaDeviceSynchronize();
		checkCudaDevice;
		dim3 threadsPerBlock(this->n, this->n, this->n);
		dim3 numBlocks(this->gridCols,this->gridRows,this->gridLevels);
		cudaDeviceSynchronize();
		checkCudaDevice;
		initRunCUDA3D<SchemeTypeHost,SchemeTypeDevice, DeviceType><<<numBlocks,threadsPerBlock,2*this->n*this->n*this->n*sizeof(double)>>>(this->cudaSolver);
		cudaDeviceSynchronize();
//		cout << "post 1 kernel" << endl;

	}
#endif


	this->currentStep = 1;
	if(this->device == tnlHostDevice)
		synchronize();
#ifdef HAVE_CUDA
	else if(this->device == tnlCudaDevice)
	{
		dim3 threadsPerBlock(this->n, this->n, this->n);
		dim3 numBlocks(this->gridCols,this->gridRows,this->gridLevels);
		//double * test = (double*)malloc(this->work_u.getSize()*sizeof(double));
		//cout << test[0] <<"   " << test[1] <<"   " << test[2] <<"   " << test[3] << endl;
		//cudaMemcpy(/*this->work_u.getData()*/ test, (this->tmpw), this->work_u.getSize()*sizeof(double), cudaMemcpyDeviceToHost);
		//cout << this->tmpw << "   " <<  test[0] <<"   " << test[1] << "   " <<test[2] << "   " <<test[3] << endl;

		checkCudaDevice;

		synchronizeCUDA3D<SchemeTypeHost, SchemeTypeDevice, DeviceType><<<numBlocks,threadsPerBlock>>>(this->cudaSolver);
		cout << cudaGetErrorString(cudaDeviceSynchronize()) << endl;
		checkCudaDevice;
		synchronize2CUDA3D<SchemeTypeHost, SchemeTypeDevice, DeviceType><<<numBlocks,1>>>(this->cudaSolver);
		cudaDeviceSynchronize();
		checkCudaDevice;
		//cout << test[0] << "   " <<test[1] <<"   " << test[2] << "   " <<test[3] << endl;
		//cudaMemcpy(/*this->work_u.getData()*/ test, (this->tmpw), this->work_u.getSize()*sizeof(double), cudaMemcpyDeviceToHost);
		//checkCudaDevice;
		//cout << this->tmpw << "   " <<  test[0] << "   " <<test[1] << "   " <<test[2] <<"   " << test[3] << endl;
		//free(test);

	}

#endif
	cout << "Solver initialized." << endl;

	return true;
}

template< typename SchemeHost, typename SchemeDevice, typename Device>
void tnlParallelEikonalSolver<3,SchemeHost, SchemeDevice, Device, double, int>::run()
{
	if(this->device == tnlHostDevice)
	{

	bool end = false;
	while ((this->boundaryConditions.max() > 0 ) || !end)
	{
		if(this->boundaryConditions.max() == 0 )
			end=true;
		else
			end=false;
#ifdef HAVE_OPENMP
#pragma omp parallel for num_threads(4) schedule(dynamic)
#endif
		for(int i = 0; i < this->subgridValues.getSize(); i++)
		{
			VectorType tmp;
			tmp.setSize(this->n*this->n*this->n);
			if(getSubgridValue(i) != INT_MAX)
			{
				//cout << "subMesh: " << i << ", BC: " << getBoundaryCondition(i) << endl;

				if(getSubgridValue(i) == currentStep+4)
				{

					if(getBoundaryCondition(i) & 1)
					{
						tmp = getSubgrid(i);
						tmp = runSubgrid(1, tmp ,i);
						insertSubgrid( tmp, i);
						this->calculationsCount[i]++;
					}
					if(getBoundaryCondition(i) & 2)
					{
						tmp = getSubgrid(i);
						tmp = runSubgrid(2, tmp ,i);
						insertSubgrid( tmp, i);
						this->calculationsCount[i]++;
					}
					if(getBoundaryCondition(i) & 4)
					{
						tmp = getSubgrid(i);
						tmp = runSubgrid(4, tmp ,i);
						insertSubgrid( tmp, i);
						this->calculationsCount[i]++;
					}
					if(getBoundaryCondition(i) & 8)
					{
						tmp = getSubgrid(i);
						tmp = runSubgrid(8, tmp ,i);
						insertSubgrid( tmp, i);
						this->calculationsCount[i]++;
					}
					if(getBoundaryCondition(i) & 16)
					{
						tmp = getSubgrid(i);
						tmp = runSubgrid(16, tmp ,i);
						insertSubgrid( tmp, i);
						this->calculationsCount[i]++;
					}
					if(getBoundaryCondition(i) & 32)
					{
						tmp = getSubgrid(i);
						tmp = runSubgrid(32, tmp ,i);
						insertSubgrid( tmp, i);
						this->calculationsCount[i]++;
					}
				}

				if( getBoundaryCondition(i) & 19)
				{
					tmp = getSubgrid(i);
					tmp = runSubgrid(19, tmp ,i);
					insertSubgrid( tmp, i);
				}
				if( getBoundaryCondition(i) & 21)
				{
					tmp = getSubgrid(i);
					tmp = runSubgrid(21, tmp ,i);
					insertSubgrid( tmp, i);
				}
				if( getBoundaryCondition(i) & 26)
				{
					tmp = getSubgrid(i);
					tmp = runSubgrid(26, tmp ,i);
					insertSubgrid( tmp, i);
				}
				if( getBoundaryCondition(i) & 28)
				{
					tmp = getSubgrid(i);
					tmp = runSubgrid(28, tmp ,i);
					insertSubgrid( tmp, i);
				}

				if( getBoundaryCondition(i) & 35)
				{
					tmp = getSubgrid(i);
					tmp = runSubgrid(35, tmp ,i);
					insertSubgrid( tmp, i);
				}
				if( getBoundaryCondition(i) & 37)
				{
					tmp = getSubgrid(i);
					tmp = runSubgrid(37, tmp ,i);
					insertSubgrid( tmp, i);
				}
				if( getBoundaryCondition(i) & 42)
				{
					tmp = getSubgrid(i);
					tmp = runSubgrid(42, tmp ,i);
					insertSubgrid( tmp, i);
				}
				if( getBoundaryCondition(i) & 44)
				{
					tmp = getSubgrid(i);
					tmp = runSubgrid(44, tmp ,i);
					insertSubgrid( tmp, i);
				}


				setBoundaryCondition(i, 0);
				setSubgridValue(i, getSubgridValue(i)-1);

			}
		}
		synchronize();
	}
	}
#ifdef HAVE_CUDA
	else if(this->device == tnlCudaDevice)
	{
		//cout << "fn" << endl;
		bool end_cuda = false;
		dim3 threadsPerBlock(this->n, this->n, this->n);
		dim3 numBlocks(this->gridCols,this->gridRows,this->gridLevels);
		cudaDeviceSynchronize();
		checkCudaDevice;
		//cudaMalloc(&runcuda,sizeof(bool));
		//cudaMemcpy(runcuda, &run_host, sizeof(bool), cudaMemcpyHostToDevice);
		//cout << "fn" << endl;
		bool* tmpb;
		//cudaMemcpy(tmpb, &(cudaSolver->runcuda),sizeof(bool*), cudaMemcpyDeviceToHost);
		//cudaDeviceSynchronize();
		//checkCudaDevice;
		cudaMemcpy(&(this->run_host),this->runcuda,sizeof(int), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		checkCudaDevice;
		//cout << "fn" << endl;
		int i = 1;
		time_diff = 0.0;
		while (run_host || !end_cuda)
		{
			cout << "Computing at step "<< i++ << endl;
			if(run_host != 0 )
				end_cuda = true;
			else
				end_cuda = false;
			//cout << "a" << endl;
			cudaDeviceSynchronize();
			checkCudaDevice;
			start = std::clock();
			runCUDA3D<SchemeTypeHost, SchemeTypeDevice, DeviceType><<<numBlocks,threadsPerBlock,2*this->n*this->n*this->n*sizeof(double)>>>(this->cudaSolver);
			//cout << "a" << endl;
			cudaDeviceSynchronize();
			time_diff += (std::clock() - start) / (double)(CLOCKS_PER_SEC);

			//start = std::clock();
			synchronizeCUDA3D<SchemeTypeHost, SchemeTypeDevice, DeviceType><<<numBlocks,threadsPerBlock>>>(this->cudaSolver);
			cudaDeviceSynchronize();
			checkCudaDevice;
			synchronize2CUDA3D<SchemeTypeHost, SchemeTypeDevice, DeviceType><<<numBlocks,1>>>(this->cudaSolver);
			cudaDeviceSynchronize();
			checkCudaDevice;
			//time_diff += (std::clock() - start) / (double)(CLOCKS_PER_SEC);


			//cout << "a" << endl;
			//run_host = false;
			//cout << "in kernel loop" << run_host << endl;
			//cudaMemcpy(tmpb, &(cudaSolver->runcuda),sizeof(bool*), cudaMemcpyDeviceToHost);
			cudaMemcpy(&run_host, (this->runcuda),sizeof(int), cudaMemcpyDeviceToHost);
			//cout << "in kernel loop" << run_host << endl;
		}
		cout << "Solving time was: " << time_diff << endl;
		//cout << "b" << endl;

		//double* tmpu;
		//cudaMemcpy(tmpu, &(cudaSolver->work_u_cuda),sizeof(double*), cudaMemcpyHostToDevice);
		//cudaMemcpy(this->work_u.getData(), tmpu, this->work_u.getSize()*sizeof(double), cudaMemcpyDeviceToHost);
		//cout << this->work_u.getData()[0] << endl;

		//double * test = (double*)malloc(this->work_u.getSize()*sizeof(double));
		//cout << test[0] << test[1] << test[2] << test[3] << endl;
		cudaMemcpy(this->work_u.getData()/* test*/, (this->tmpw), this->work_u.getSize()*sizeof(double), cudaMemcpyDeviceToHost);
		//cout << this->tmpw << "   " <<  test[0] << test[1] << test[2] << test[3] << endl;
		//free(test);

		cudaDeviceSynchronize();
	}
#endif
	contractGrid();
	this->u0.save("u-00001.tnl");
	cout << "Maximum number of calculations on one subgrid was " << this->calculationsCount.absMax() << endl;
	cout << "Average number of calculations on one subgrid was " << ( (double) this->calculationsCount.sum() / (double) this->calculationsCount.getSize() ) << endl;
	cout << "Solver finished" << endl;

#ifdef HAVE_CUDA
	if(this->device == tnlCudaDevice)
	{
		cudaFree(this->runcuda);
		cudaFree(this->tmpw);
		cudaFree(this->cudaSolver);
	}
#endif

}

//north - 1, east - 2, west - 4, south - 8
template< typename SchemeHost, typename SchemeDevice, typename Device>
void tnlParallelEikonalSolver<3,SchemeHost, SchemeDevice, Device, double, int>::synchronize() //needs fix ---- maybe not anymore --- but frankly: yeah, it does -- aaaa-and maybe fixed now
{
	cout << "Synchronizig..." << endl;
	int tmp1, tmp2;
	int grid1, grid2;

//	if(this->currentStep & 1)
//	{
		for(int j = 0; j < this->gridRows - 1; j++)
		{
			for (int i = 0; i < this->gridCols*this->n; i++)
			{
				for (int k = 0; k < this->gridLevels*this->n; k++)
				{
//					cout << "a" << endl;
					tmp1 = this->gridCols*this->n*((this->n-1)+j*this->n) + i + k*this->gridCols*this->n*this->gridRows*this->n;
//					cout << "b" << endl;
					tmp2 = this->gridCols*this->n*((this->n)+j*this->n) + i + k*this->gridCols*this->n*this->gridRows*this->n;
//					cout << "c" << endl;
					if(tmp1 > work_u.getSize())
						cout << "tmp1: " << tmp1 << " x: " << j <<" y: " << i <<" z: " << k << endl;
					if(tmp2 > work_u.getSize())
						cout << "tmp2: " << tmp2 << " x: " << j <<" y: " << i <<" z: " << k << endl;
					grid1 = getSubgridValue(getOwner(tmp1));
//					cout << "d" << endl;
					grid2 = getSubgridValue(getOwner(tmp2));
//					cout << "e" << endl;
					if(getOwner(tmp1)==getOwner(tmp2))
						cout << "i, j, k" << i << "," << j << "," << k << endl;
					if ((fabs(this->work_u[tmp1]) < fabs(this->work_u[tmp2]) - this->delta || grid2 == INT_MAX || grid2 == -INT_MAX) && (grid1 != INT_MAX && grid1 != -INT_MAX))
					{
						this->work_u[tmp2] = this->work_u[tmp1];
//						cout << "f" << endl;
						this->unusedCell[tmp2] = 0;
//						cout << "g" << endl;
						if(grid2 == INT_MAX)
						{
							setSubgridValue(getOwner(tmp2), -INT_MAX);
						}
//						cout << "h" << endl;
						if(! (getBoundaryCondition(getOwner(tmp2)) & 8) )
							setBoundaryCondition(getOwner(tmp2), getBoundaryCondition(getOwner(tmp2))+8);
//						cout << "i" << endl;
					}
					else if ((fabs(this->work_u[tmp1]) > fabs(this->work_u[tmp2]) + this->delta || grid1 == INT_MAX || grid1 == -INT_MAX) && (grid2 != INT_MAX && grid2 != -INT_MAX))
					{
						this->work_u[tmp1] = this->work_u[tmp2];
//						cout << "j" << endl;
						this->unusedCell[tmp1] = 0;
//						cout << "k" << endl;
						if(grid1 == INT_MAX)
						{
							setSubgridValue(getOwner(tmp1), -INT_MAX);
						}
//						cout << "l" << endl;
						if(! (getBoundaryCondition(getOwner(tmp1)) & 1) )
							setBoundaryCondition(getOwner(tmp1), getBoundaryCondition(getOwner(tmp1))+1);
//						cout << "m" << endl;
					}
				}
			}
		}

//	}
//	else
//	{

		cout << "sync 2" << endl;
		for(int i = 1; i < this->gridCols; i++)
		{
			for (int j = 0; j < this->gridRows*this->n; j++)
			{
				for (int k = 0; k < this->gridLevels*this->n; k++)
				{
					tmp1 = this->gridCols*this->n*j + i*this->n - 1 + k*this->gridCols*this->n*this->gridRows*this->n;
					tmp2 = this->gridCols*this->n*j + i*this->n + k*this->gridCols*this->n*this->gridRows*this->n;
					grid1 = getSubgridValue(getOwner(tmp1));
					grid2 = getSubgridValue(getOwner(tmp2));
					if(getOwner(tmp1)==getOwner(tmp2))
						cout << "i, j, k" << i << "," << j << "," << k << endl;
					if ((fabs(this->work_u[tmp1]) < fabs(this->work_u[tmp2]) - this->delta || grid2 == INT_MAX || grid2 == -INT_MAX) && (grid1 != INT_MAX && grid1 != -INT_MAX))
					{
						this->work_u[tmp2] = this->work_u[tmp1];
						this->unusedCell[tmp2] = 0;
						if(grid2 == INT_MAX)
						{
							setSubgridValue(getOwner(tmp2), -INT_MAX);
						}
						if(! (getBoundaryCondition(getOwner(tmp2)) & 4) )
							setBoundaryCondition(getOwner(tmp2), getBoundaryCondition(getOwner(tmp2))+4);
					}
					else if ((fabs(this->work_u[tmp1]) > fabs(this->work_u[tmp2]) + this->delta || grid1 == INT_MAX || grid1 == -INT_MAX) && (grid2 != INT_MAX && grid2 != -INT_MAX))
					{
						this->work_u[tmp1] = this->work_u[tmp2];
						this->unusedCell[tmp1] = 0;
						if(grid1 == INT_MAX)
						{
							setSubgridValue(getOwner(tmp1), -INT_MAX);
						}
						if(! (getBoundaryCondition(getOwner(tmp1)) & 2) )
							setBoundaryCondition(getOwner(tmp1), getBoundaryCondition(getOwner(tmp1))+2);
					}
				}
			}
		}

		cout << "sync 3" << endl;

		for(int k = 1; k < this->gridLevels; k++)
		{
			for (int j = 0; j < this->gridRows*this->n; j++)
			{
				for (int i = 0; i < this->gridCols*this->n; i++)
				{
					tmp1 = this->gridCols*this->n*j + i + (k*this->n-1)*this->gridCols*this->n*this->gridRows*this->n;
					tmp2 = this->gridCols*this->n*j + i + k*this->n*this->gridCols*this->n*this->gridRows*this->n;
					grid1 = getSubgridValue(getOwner(tmp1));
					grid2 = getSubgridValue(getOwner(tmp2));
					if(getOwner(tmp1)==getOwner(tmp2))
						cout << "i, j, k" << i << "," << j << "," << k << endl;
					if ((fabs(this->work_u[tmp1]) < fabs(this->work_u[tmp2]) - this->delta || grid2 == INT_MAX || grid2 == -INT_MAX) && (grid1 != INT_MAX && grid1 != -INT_MAX))
					{
						this->work_u[tmp2] = this->work_u[tmp1];
						this->unusedCell[tmp2] = 0;
						if(grid2 == INT_MAX)
						{
							setSubgridValue(getOwner(tmp2), -INT_MAX);
						}
						if(! (getBoundaryCondition(getOwner(tmp2)) & 32) )
							setBoundaryCondition(getOwner(tmp2), getBoundaryCondition(getOwner(tmp2))+32);
					}
					else if ((fabs(this->work_u[tmp1]) > fabs(this->work_u[tmp2]) + this->delta || grid1 == INT_MAX || grid1 == -INT_MAX) && (grid2 != INT_MAX && grid2 != -INT_MAX))
					{
						this->work_u[tmp1] = this->work_u[tmp2];
						this->unusedCell[tmp1] = 0;
						if(grid1 == INT_MAX)
						{
							setSubgridValue(getOwner(tmp1), -INT_MAX);
						}
						if(! (getBoundaryCondition(getOwner(tmp1)) & 16) )
							setBoundaryCondition(getOwner(tmp1), getBoundaryCondition(getOwner(tmp1))+16);
					}
				}
			}
		}
//		}



	this->currentStep++;
	int stepValue = this->currentStep + 4;
	for (int i = 0; i < this->subgridValues.getSize(); i++)
	{
		if( getSubgridValue(i) == -INT_MAX )
			setSubgridValue(i, stepValue);
	}

	cout << "Grid synchronized at step " << (this->currentStep - 1 ) << endl;

}


template< typename SchemeHost, typename SchemeDevice, typename Device>
int tnlParallelEikonalSolver<3,SchemeHost, SchemeDevice, Device, double, int>::getOwner(int i) const
{

	int j = i % (this->gridCols*this->gridRows*this->n*this->n);

	return ( (i / (this->gridCols*this->gridRows*this->n*this->n*this->n))*this->gridCols*this->gridRows
			+ (j / (this->gridCols*this->n*this->n))*this->gridCols
			+ (j % (this->gridCols*this->n))/this->n);
}

template< typename SchemeHost, typename SchemeDevice, typename Device>
int tnlParallelEikonalSolver<3,SchemeHost, SchemeDevice, Device, double, int>::getSubgridValue( int i ) const
{
	return this->subgridValues[i];
}

template< typename SchemeHost, typename SchemeDevice, typename Device>
void tnlParallelEikonalSolver<3,SchemeHost, SchemeDevice, Device, double, int>::setSubgridValue(int i, int value)
{
	this->subgridValues[i] = value;
}

template< typename SchemeHost, typename SchemeDevice, typename Device>
int tnlParallelEikonalSolver<3,SchemeHost, SchemeDevice, Device, double, int>::getBoundaryCondition( int i ) const
{
	return this->boundaryConditions[i];
}

template< typename SchemeHost, typename SchemeDevice, typename Device>
void tnlParallelEikonalSolver<3,SchemeHost, SchemeDevice, Device, double, int>::setBoundaryCondition(int i, int value)
{
	this->boundaryConditions[i] = value;
}

template< typename SchemeHost, typename SchemeDevice, typename Device>
void tnlParallelEikonalSolver<3,SchemeHost, SchemeDevice, Device, double, int>::stretchGrid()
{
	cout << "Stretching grid..." << endl;


	this->gridCols = ceil( ((double)(this->mesh.getDimensions().x()-1)) / ((double)(this->n-1)) );
	this->gridRows = ceil( ((double)(this->mesh.getDimensions().y()-1)) / ((double)(this->n-1)) );
	this->gridLevels = ceil( ((double)(this->mesh.getDimensions().z()-1)) / ((double)(this->n-1)) );

	//this->gridCols = (this->mesh.getDimensions().x()-1) / (this->n-1) ;
	//this->gridRows = (this->mesh.getDimensions().y()-1) / (this->n-1) ;

	cout << "Setting gridCols to " << this->gridCols << "." << endl;
	cout << "Setting gridRows to " << this->gridRows << "." << endl;
	cout << "Setting gridLevels to " << this->gridLevels << "." << endl;

	this->subgridValues.setSize(this->gridCols*this->gridRows*this->gridLevels);
	this->subgridValues.setValue(0);
	this->boundaryConditions.setSize(this->gridCols*this->gridRows*this->gridLevels);
	this->boundaryConditions.setValue(0);
	this->calculationsCount.setSize(this->gridCols*this->gridRows*this->gridLevels);
	this->calculationsCount.setValue(0);

	for(int i = 0; i < this->subgridValues.getSize(); i++ )
	{
		this->subgridValues[i] = INT_MAX;
		this->boundaryConditions[i] = 0;
	}

	int levelSize = this->n*this->n*this->gridCols*this->gridRows;
	int stretchedSize = this->n*levelSize*this->gridLevels;

	if(!this->work_u.setSize(stretchedSize))
		cerr << "Could not allocate memory for stretched grid." << endl;
	if(!this->unusedCell.setSize(stretchedSize))
		cerr << "Could not allocate memory for supporting stretched grid." << endl;
	int idealStretch =this->mesh.getDimensions().x() + (this->mesh.getDimensions().x()-2)/(this->n-1);
	cout << idealStretch << endl;




	for(int i = 0; i < levelSize; i++)
	{
		int diff =(this->n*this->gridCols) - idealStretch ;

		int k = i/this->n - i/(this->n*this->gridCols) + this->mesh.getDimensions().x()*(i/(this->n*this->n*this->gridCols)) + (i/(this->n*this->gridCols))*diff;

		if(i%(this->n*this->gridCols) - idealStretch  >= 0)
		{
			k+= i%(this->n*this->gridCols) - idealStretch +1 ;
		}

		if(i/(this->n*this->gridCols) - idealStretch + 1  > 0)
		{
			k+= (i/(this->n*this->gridCols) - idealStretch +1 )* this->mesh.getDimensions().x() ;
		}

		for( int j = 0; j<this->n*this->gridLevels; j++)
		{
			this->unusedCell[i+j*levelSize] = 1;
			int l = j/this->n;

			if(j - idealStretch  >= 0)
			{
				l+= j - idealStretch + 1;
			}

			this->work_u[i+j*levelSize] = this->u0[i+(j-l)*mesh.getDimensions().x()*mesh.getDimensions().y()-k];
		}

	}



	cout << "Grid stretched." << endl;
}

template< typename SchemeHost, typename SchemeDevice, typename Device>
void tnlParallelEikonalSolver<3,SchemeHost, SchemeDevice, Device, double, int>::contractGrid()
{
	cout << "Contracting grid..." << endl;
	int levelSize = this->n*this->n*this->gridCols*this->gridRows;
	int stretchedSize = this->n*levelSize*this->gridLevels;

	int idealStretch =this->mesh.getDimensions().x() + (this->mesh.getDimensions().x()-2)/(this->n-1);
	cout << idealStretch << endl;


	for(int i = 0; i < levelSize; i++)
	{
		int diff =(this->n*this->gridCols) - idealStretch ;
		int k = i/this->n - i/(this->n*this->gridCols) + this->mesh.getDimensions().x()*(i/(this->n*this->n*this->gridCols)) + (i/(this->n*this->gridCols))*diff;

		if((i%(this->n*this->gridCols) - idealStretch  < 0) && (i/(this->n*this->gridCols) - idealStretch + 1  <= 0) )
		{
			for( int j = 0; j<this->n*this->gridLevels; j++)
			{
				int l = j/this->n;
				if(j - idealStretch  < 0)
					this->u0[i+(j-l)*mesh.getDimensions().x()*mesh.getDimensions().y()-k] = this->work_u[i+j*levelSize];
			}
		}

	}

	cout << "Grid contracted" << endl;
}

template< typename SchemeHost, typename SchemeDevice, typename Device>
typename tnlParallelEikonalSolver<3,SchemeHost, SchemeDevice, Device, double, int>::VectorType
tnlParallelEikonalSolver<3,SchemeHost, SchemeDevice, Device, double, int>::getSubgrid( const int i ) const
{

	VectorType u;
	u.setSize(this->n*this->n*this->n);

	int idx, idy, idz;
	idz = i / (gridRows*this->gridCols);
	idy = (i % (this->gridRows*this->gridCols)) / this->gridCols;
	idx = i %  (this->gridCols);

	for( int j = 0; j < this->n; j++)
	{
	//	int index = (i / this->gridCols)*this->n*this->n*this->gridCols + (i % this->gridCols)*this->n + (j/this->n)*this->n*this->gridCols + (j % this->n);
		for( int k = 0; k < this->n; k++)
		{
			for( int l = 0; l < this->n; l++)
			{
				int index = (idz*this->n + l) * this->n*this->n*this->gridCols*this->gridRows
						 + (idy) * this->n*this->n*this->gridCols
						 + (idx) * this->n
						 + k * this->n*this->gridCols
						 + j;

				u[j + k*this->n  + l*this->n*this->n] = this->work_u[ index ];
			}
		}
	}
	return u;
}

template< typename SchemeHost, typename SchemeDevice, typename Device>
void tnlParallelEikonalSolver<3,SchemeHost, SchemeDevice, Device, double, int>::insertSubgrid( VectorType u, const int i )
{
	int idx, idy, idz;
	idz = i / (this->gridRows*this->gridCols);
	idy = (i % (this->gridRows*this->gridCols)) / this->gridCols;
	idx = i %  (this->gridCols);

	for( int j = 0; j < this->n; j++)
	{
	//	int index = (i / this->gridCols)*this->n*this->n*this->gridCols + (i % this->gridCols)*this->n + (j/this->n)*this->n*this->gridCols + (j % this->n);
		for( int k = 0; k < this->n; k++)
		{
			for( int l = 0; l < this->n; l++)
			{

				int index = (idz*this->n + l) * this->n*this->n*this->gridCols*this->gridRows
						 + (idy) * this->n*this->n*this->gridCols
						 + (idx) * this->n
						 + k * this->n*this->gridCols
						 + j;

				//OMP LOCK index
//				cout<< idx << " " << idy << " " << idz << " " << j << " " << k << " " << l << " " << idz << " " << unusedCell.getSize() << " " << u.getSize() << " " << index <<endl;
				if( (fabs(this->work_u[index]) > fabs(u[j + k*this->n  + l*this->n*this->n])) || (this->unusedCell[index] == 1) )
				{
					this->work_u[index] = u[j + k*this->n  + l*this->n*this->n];
					this->unusedCell[index] = 0;
				}
				//OMP UNLOCK index
			}
		}
	}
}

template< typename SchemeHost, typename SchemeDevice, typename Device>
typename tnlParallelEikonalSolver<3,SchemeHost, SchemeDevice, Device, double, int>::VectorType
tnlParallelEikonalSolver<3,SchemeHost, SchemeDevice, Device, double, int>::runSubgrid( int boundaryCondition, VectorType u, int subGridID)
{

	VectorType fu;

	fu.setLike(u);
	fu.setValue( 0.0 );


	bool tmp = false;
	for(int i = 0; i < u.getSize(); i++)
	{
		if(u[0]*u[i] <= 0.0)
			tmp=true;
	}
	int idx,idy,idz;
	idz = subGridID / (this->gridRows*this->gridCols);
	idy = (subGridID % (this->gridRows*this->gridCols)) / this->gridCols;
	idx = subGridID %  (this->gridCols);
	int centreGID = (this->n*idy + (this->n>>1) )*(this->n*this->gridCols) + this->n*idx + (this->n>>1)
			      + ((this->n>>1)+this->n*idz)*this->n*this->n*this->gridRows*this->gridCols;
	if(this->unusedCell[centreGID] == 0 || boundaryCondition == 0)
		tmp = true;
	//if(this->currentStep + 3 < getSubgridValue(subGridID))
		//tmp = true;


	double value = Sign(u[0]) * u.absMax();

	if(tmp)
	{}


	//north - 1, east - 2, west - 4, south - 8
	else if(boundaryCondition == 4)
	{
		for(int i = 0; i < this->n; i++)
			for(int j = 1;j < this->n; j++)
				for(int k = 0;k < this->n; k++)
				//if(fabs(u[i*this->n + j]) <  fabs(u[i*this->n]))
				u[k*this->n*this->n + i*this->n + j] = value;// u[i*this->n];
	}
	else if(boundaryCondition == 2)
	{
		for(int i = 0; i < this->n; i++)
			for(int j =0 ;j < this->n -1; j++)
				for(int k = 0;k < this->n; k++)
				//if(fabs(u[i*this->n + j]) < fabs(u[(i+1)*this->n - 1]))
				u[k*this->n*this->n + i*this->n + j] = value;// u[(i+1)*this->n - 1];
	}
	else if(boundaryCondition == 1)
	{
		for(int j = 0; j < this->n; j++)
			for(int i = 0;i < this->n - 1; i++)
				for(int k = 0;k < this->n; k++)
				//if(fabs(u[i*this->n + j]) < fabs(u[j + this->n*(this->n - 1)]))
				u[k*this->n*this->n + i*this->n + j] = value;// u[j + this->n*(this->n - 1)];
	}
	else if(boundaryCondition == 8)
	{
		for(int j = 0; j < this->n; j++)
			for(int i = 1;i < this->n; i++)
				for(int k = 0;k < this->n; k++)
				//if(fabs(u[i*this->n + j]) < fabs(u[j]))
				u[k*this->n*this->n + i*this->n + j] = value;// u[j];
	}
	else if(boundaryCondition == 16)
	{
		for(int j = 0; j < this->n; j++)
			for(int i = 0;i < this->n ; i++)
				for(int k = 0;k < this->n-1; k++)
				//if(fabs(u[i*this->n + j]) < fabs(u[j + this->n*(this->n - 1)]))
				u[k*this->n*this->n + i*this->n + j] = value;// u[j + this->n*(this->n - 1)];
	}
	else if(boundaryCondition == 32)
	{
		for(int j = 0; j < this->n; j++)
			for(int i = 0;i < this->n; i++)
				for(int k = 1;k < this->n; k++)
				//if(fabs(u[i*this->n + j]) < fabs(u[j]))
				u[k*this->n*this->n + i*this->n + j] = value;// u[j];
	}


   double time = 0.0;
   double currentTau = this->tau0;
   double finalTime = this->stopTime;// + 3.0*(u.max() - u.min());
   if(boundaryCondition == 0) finalTime *= 2.0;
   if( time + currentTau > finalTime ) currentTau = finalTime - time;

   double maxResidue( 1.0 );
   //double lastResidue( 10000.0 );
   tnlGridEntity<MeshType, 3, tnlGridEntityNoStencilStorage > Entity(subMesh);
   tnlNeighbourGridEntityGetter<tnlGridEntity< MeshType, 3, tnlGridEntityNoStencilStorage >,3> neighbourEntities(Entity);
   while( time < finalTime /*|| maxResidue > subMesh.template getSpaceStepsProducts< 1, 0, 0 >()*/)
   {
      /****
       * Compute the RHS
       */

      for( int i = 0; i < fu.getSize(); i ++ )
      {
//    	  cout << "i: " << i << ", time: " << time <<endl;
    	  tnlStaticVector<3,int> coords(i % subMesh.getDimensions().x(),
    	  								(i % (subMesh.getDimensions().x()*subMesh.getDimensions().y())) / subMesh.getDimensions().x(),
    	  								i / (subMesh.getDimensions().x()*subMesh.getDimensions().y()));
//    	  	cout << "b " << i << " " << i % subMesh.getDimensions().x() << " " << (i % (subMesh.getDimensions().x()*subMesh.getDimensions().y())) << " " << (i % subMesh.getDimensions().x()*subMesh.getDimensions().y()) / subMesh.getDimensions().x() << " " << subMesh.getDimensions().x()*subMesh.getDimensions().y() << " " <<endl;
			Entity.setCoordinates(coords);
//			cout <<"c" << coords << endl;
			Entity.refresh();
//			cout << "d" <<endl;
			neighbourEntities.refresh(subMesh,Entity.getIndex());
//			cout << "e" <<endl;
    	  fu[ i ] = schemeHost.getValue( this->subMesh, i, coords,u, time, boundaryCondition, neighbourEntities );
//    	  cout << "f" <<endl;
      }
      maxResidue = fu. absMax();


      if( this -> cflCondition * maxResidue != 0.0)
    	  currentTau =  this -> cflCondition / maxResidue;

     /* if (maxResidue < 0.05)
    	  cout << "Max < 0.05" << endl;*/
      if(currentTau > 0.5 * this->subMesh.template getSpaceStepsProducts< 1, 0, 0 >())
    	  currentTau = 0.5 * this->subMesh.template getSpaceStepsProducts< 1, 0, 0 >();
      /*if(maxResidue > lastResidue)
    	  currentTau *=(1.0/10.0);*/


      if( time + currentTau > finalTime ) currentTau = finalTime - time;
//      for( int i = 0; i < fu.getSize(); i ++ )
//      {
//    	  //cout << "Too big RHS! i = " << i << ", fu = " << fu[i] << ", u = " << u[i] << endl;
//    	  if((u[i]+currentTau * fu[ i ])*u[i] < 0.0 && fu[i] != 0.0 && u[i] != 0.0 )
//    		  currentTau = fabs(u[i]/(2.0*fu[i]));
//
//      }


      for( int i = 0; i < fu.getSize(); i ++ )
      {
    	  double add = u[i] + currentTau * fu[ i ];
    	  //if( fabs(u[i]) < fabs(add) or (this->subgridValues[subGridID] == this->currentStep +4) )
    		  u[ i ] = add;
      }
      time += currentTau;

      //cout << '\r' << flush;
     //cout << maxResidue << "   " << currentTau << " @ " << time << flush;
     //lastResidue = maxResidue;
   }
   //cout << "Time: " << time << ", Res: " << maxResidue <<endl;
	/*if (u.max() > 0.0)
		this->stopTime /=(double) this->gridCols;*/

//	VectorType solution;
//	solution.setLike(u);
//    for( int i = 0; i < u.getSize(); i ++ )
//  	{
//    	solution[i]=u[i];
//   	}
//	return solution;
	return u;
}


#ifdef HAVE_CUDA


template< typename SchemeHost, typename SchemeDevice, typename Device>
__device__
void tnlParallelEikonalSolver<3,SchemeHost, SchemeDevice, Device, double, int>::getSubgridCUDA3D( const int i ,tnlParallelEikonalSolver<3,SchemeHost, SchemeDevice, Device, double, int >* caller, double* a)
{
	//int j = threadIdx.x + threadIdx.y * blockDim.x;
	int index = (blockIdx.z*this->n + threadIdx.z) * this->n*this->n*this->gridCols*this->gridRows
			 + (blockIdx.y) * this->n*this->n*this->gridCols
             + (blockIdx.x) * this->n
             + threadIdx.y * this->n*this->gridCols
             + threadIdx.x;
	//printf("i= %d,j= %d,th= %d\n",i,j,th);
	*a = caller->work_u_cuda[index];
	//printf("Hi %f \n", *a);
	//return ret;
}


template< typename SchemeHost, typename SchemeDevice, typename Device>
__device__
void tnlParallelEikonalSolver<3,SchemeHost, SchemeDevice, Device, double, int>::updateSubgridCUDA3D( const int i ,tnlParallelEikonalSolver<3,SchemeHost, SchemeDevice, Device, double, int >* caller, double* a)
{
//	int j = threadIdx.x + threadIdx.y * blockDim.x;
	int index = (blockIdx.z*this->n + threadIdx.z) * this->n*this->n*this->gridCols*this->gridRows
			 + (blockIdx.y) * this->n*this->n*this->gridCols
             + (blockIdx.x) * this->n
             + threadIdx.y * this->n*this->gridCols
             + threadIdx.x;

	if( (fabs(caller->work_u_cuda[index]) > fabs(*a)) || (caller->unusedCell_cuda[index] == 1) )
	{
		caller->work_u_cuda[index] = *a;
		caller->unusedCell_cuda[index] = 0;

	}

	*a = caller->work_u_cuda[index];
}


template< typename SchemeHost, typename SchemeDevice, typename Device>
__device__
void tnlParallelEikonalSolver<3,SchemeHost, SchemeDevice, Device, double, int>::insertSubgridCUDA3D( double u, const int i )
{


//	int j = threadIdx.x + threadIdx.y * blockDim.x;
	//printf("j = %d, u = %f\n", j,u);

		int index = (blockIdx.z*this->n + threadIdx.z) * this->n*this->n*this->gridCols*this->gridRows
				 + (blockIdx.y) * this->n*this->n*this->gridCols
	             + (blockIdx.x) * this->n
	             + threadIdx.y * this->n*this->gridCols
	             + threadIdx.x;

		//printf("i= %d,j= %d,index= %d\n",i,j,index);
		if( (fabs(this->work_u_cuda[index]) > fabs(u)) || (this->unusedCell_cuda[index] == 1) )
		{
			this->work_u_cuda[index] = u;
			this->unusedCell_cuda[index] = 0;

		}


}


template< typename SchemeHost, typename SchemeDevice, typename Device>
__device__
void tnlParallelEikonalSolver<3,SchemeHost, SchemeDevice, Device, double, int>::runSubgridCUDA3D( int boundaryCondition, double* u, int subGridID)
{

	__shared__ int tmp;
	__shared__ double value;
	//double tmpRes = 0.0;
	volatile double* sharedTau = &u[blockDim.x*blockDim.y*blockDim.z];
//	volatile double* absVal = &u[2*blockDim.x*blockDim.y*blockDim.z];
	int i = threadIdx.x;
	int j = threadIdx.y;
	int k = threadIdx.z;
	int l = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
	bool computeFU = !((i == 0 && (boundaryCondition & 4)) or
			 (i == blockDim.x - 1 && (boundaryCondition & 2)) or
			 (j == 0 && (boundaryCondition & 8)) or
			 (j == blockDim.y - 1  && (boundaryCondition & 1))or
			 (k == 0 && (boundaryCondition & 32)) or
			 (k == blockDim.z - 1  && (boundaryCondition & 16)));

	if(l == 0)
	{
		tmp = 0;
		int centreGID = (blockDim.y*blockIdx.y + (blockDim.y>>1) )*(blockDim.x*gridDim.x) + blockDim.x*blockIdx.x + (blockDim.x>>1)
				      + ((blockDim.z>>1)+blockDim.z*blockIdx.z)*blockDim.x*blockDim.y*gridDim.x*gridDim.y;
		if(this->unusedCell_cuda[centreGID] == 0 || boundaryCondition == 0)
			tmp = 1;
	}
	__syncthreads();


	__syncthreads();
	if(tmp !=1)
	{
//		if(computeFU)
//			absVal[l]=0.0;
//		else
//			absVal[l] = fabs(u[l]);
//
//		__syncthreads();
//
//	      if((blockDim.x == 16) && (l < 128))		absVal[l] = Max(absVal[l],absVal[l+128]);
//	      __syncthreads();
//	      if((blockDim.x == 16) && (l < 64))		absVal[l] = Max(absVal[l],absVal[l+64]);
//	      __syncthreads();
//	      if(l < 32)    							absVal[l] = Max(absVal[l],absVal[l+32]);
//	      if(l < 16)								absVal[l] = Max(absVal[l],absVal[l+16]);
//	      if(l < 8)									absVal[l] = Max(absVal[l],absVal[l+8]);
//	      if(l < 4)									absVal[l] = Max(absVal[l],absVal[l+4]);
//	      if(l < 2)									absVal[l] = Max(absVal[l],absVal[l+2]);
//	      if(l < 1)									value   = Sign(u[0])*Max(absVal[l],absVal[l+1]);
//		__syncthreads();
//
//		if(computeFU)
//			u[l] = value;
		if(computeFU)
		{
			tnlGridEntity<MeshType, 3, tnlGridEntityNoStencilStorage > Ent(subMesh);
			if(boundaryCondition == 4)
			{
				Ent.setCoordinates(tnlStaticVector<3,int>(0,j,k));
			   	Ent.refresh();
				u[l] = u[Ent.getIndex()];// + Sign(u[0])*this->subMesh.template getSpaceStepsProducts< 1, 0, 0 >()*(threadIdx.x) ;//+  2*Sign(u[0])*this->subMesh.template getSpaceStepsProducts< 1, 0, 0 >()*(threadIdx.x+this->n);
			}
			else if(boundaryCondition == 2)
			{
				Ent.setCoordinates(tnlStaticVector<3,int>(blockDim.x - 1,j,k));
			   	Ent.refresh();
				u[l] = u[Ent.getIndex()];// + Sign(u[0])*this->subMesh.template getSpaceStepsProducts< 1, 0, 0 >()*(this->n - 1 - threadIdx.x);//+ 2*Sign(u[0])*this->subMesh.template getSpaceStepsProducts< 1, 0, 0 >()*(blockDim.x - threadIdx.x - 1+this->n);
			}
			else if(boundaryCondition == 8)
			{
				Ent.setCoordinates(tnlStaticVector<3,int>(i,0,k));
			   	Ent.refresh();
				u[l] = u[Ent.getIndex()];// + Sign(u[0])*this->subMesh.template getSpaceStepsProducts< 0, 1, 0 >()*(threadIdx.y) ;//+ 2*Sign(u[0])*this->subMesh.template getSpaceStepsProducts< 1, 0, 0 >()*(threadIdx.y+this->n);
			}
			else if(boundaryCondition == 1)
			{
				Ent.setCoordinates(tnlStaticVector<3,int>(i,blockDim.y - 1,k));
			   	Ent.refresh();
				u[l] = u[Ent.getIndex()];// + Sign(u[0])*this->subMesh.template getSpaceStepsProducts< 0, 1, 0 >()*(this->n - 1 - threadIdx.y) ;//+ Sign(u[0])*this->subMesh.template getSpaceStepsProducts< 1, 0, 0 >()*(blockDim.y - threadIdx.y  - 1 +this->n);
			}
			else if(boundaryCondition == 32)
			{
				Ent.setCoordinates(tnlStaticVector<3,int>(i,j,0));
			   	Ent.refresh();
				u[l] = u[Ent.getIndex()];// + Sign(u[0])*this->subMesh.template getSpaceStepsProducts< 0, 0, 1 >()*(threadIdx.z);
			}
			else if(boundaryCondition == 16)
			{
				Ent.setCoordinates(tnlStaticVector<3,int>(i,j,blockDim.z - 1));
			   	Ent.refresh();
				u[l] = u[Ent.getIndex()];// + Sign(u[0])*this->subMesh.template getSpaceStepsProducts< 0, 0, 1 >()*(this->n - 1 - threadIdx.z) ;
			}
		}
	}

   double time = 0.0;
   __shared__ double currentTau;
   double cfl = this->cflCondition;
   double fu = 0.0;
//   if(threadIdx.x * threadIdx.y * threadIdx.z == 0)
//   {
//	   currentTau = this->tau0;
//   }
   double finalTime = this->stopTime;
   __syncthreads();
//   if( boundaryCondition == 0 ) finalTime *= 2.0;

   tnlGridEntity<MeshType, 3, tnlGridEntityNoStencilStorage > Entity(subMesh);
   tnlNeighbourGridEntityGetter<tnlGridEntity< MeshType, 3, tnlGridEntityNoStencilStorage >,3> neighbourEntities(Entity);
   Entity.setCoordinates(tnlStaticVector<3,int>(i,j,k));
   Entity.refresh();
   neighbourEntities.refresh(subMesh,Entity.getIndex());


   while( time < finalTime )
   {
	  sharedTau[l]=finalTime;

	  if(computeFU)
	  {
		  fu = schemeHost.getValueDev( this->subMesh, l, tnlStaticVector<3,int>(i,j,k), u, time, boundaryCondition, neighbourEntities);
		  if(abs(fu) > 0.0)
			  sharedTau[l]=abs(cfl/fu);
	  }

      if(l == 0)
      {
    	  if(sharedTau[0] > 0.5 * this->subMesh.template getSpaceStepsProducts< 1, 0, 0 >())	sharedTau[0] = 0.5 * this->subMesh.template getSpaceStepsProducts< 1, 0, 0 >();
      }
      else if(l == blockDim.x*blockDim.y*blockDim.z - 1)
      {
    	  if( time + sharedTau[l] > finalTime )		sharedTau[l] = finalTime - time;
      }


      if(l < 256)								sharedTau[l] = Min(sharedTau[l],sharedTau[l+256]);
      __syncthreads();
      if(l < 128)								sharedTau[l] = Min(sharedTau[l],sharedTau[l+128]);
      __syncthreads();
      if(l < 64)								sharedTau[l] = Min(sharedTau[l],sharedTau[l+64]);
      __syncthreads();
      if(l < 32)    							sharedTau[l] = Min(sharedTau[l],sharedTau[l+32]);
      __syncthreads();
      if(l < 16)								sharedTau[l] = Min(sharedTau[l],sharedTau[l+16]);
      if(l < 8)									sharedTau[l] = Min(sharedTau[l],sharedTau[l+8]);
      if(l < 4)									sharedTau[l] = Min(sharedTau[l],sharedTau[l+4]);
      if(l < 2)									sharedTau[l] = Min(sharedTau[l],sharedTau[l+2]);
      if(l < 1)									currentTau   = Min(sharedTau[l],sharedTau[l+1]);
      __syncthreads();

//	if(abs(fu) < 10000.0)
//		printf("bla");
      if(computeFU)
    	  u[l] += currentTau * fu;
      time += currentTau;
      __syncthreads();
   }


}


template< typename SchemeHost, typename SchemeDevice, typename Device>
__device__
int tnlParallelEikonalSolver<3,SchemeHost, SchemeDevice, Device, double, int>::getOwnerCUDA3D(int i) const
{
	int j = i % (this->gridCols*this->gridRows*this->n*this->n);

	return ( (i / (this->gridCols*this->gridRows*this->n*this->n))*this->gridCols*this->gridRows
			+ (j / (this->gridCols*this->n*this->n))*this->gridCols
			+ (j % (this->gridCols*this->n))/this->n);
}


template< typename SchemeHost, typename SchemeDevice, typename Device>
__device__
int tnlParallelEikonalSolver<3,SchemeHost, SchemeDevice, Device, double, int>::getSubgridValueCUDA3D( int i ) const
{
	return this->subgridValues_cuda[i];
}


template< typename SchemeHost, typename SchemeDevice, typename Device>
__device__
void tnlParallelEikonalSolver<3,SchemeHost, SchemeDevice, Device, double, int>::setSubgridValueCUDA3D(int i, int value)
{
	this->subgridValues_cuda[i] = value;
}


template< typename SchemeHost, typename SchemeDevice, typename Device>
__device__
int tnlParallelEikonalSolver<3,SchemeHost, SchemeDevice, Device, double, int>::getBoundaryConditionCUDA3D( int i ) const
{
	return this->boundaryConditions_cuda[i];
}


template< typename SchemeHost, typename SchemeDevice, typename Device>
__device__
void tnlParallelEikonalSolver<3,SchemeHost, SchemeDevice, Device, double, int>::setBoundaryConditionCUDA3D(int i, int value)
{
	this->boundaryConditions_cuda[i] = value;
}



//north - 1, east - 2, west - 4, south - 8, up -16, down - 32

template <typename SchemeHost, typename SchemeDevice, typename Device>
__global__
void /*tnlParallelEikonalSolver<3,SchemeHost, SchemeDevice, Device, double, int>::*/synchronizeCUDA3D(tnlParallelEikonalSolver<3,SchemeHost, SchemeDevice, Device, double, int >* cudaSolver) //needs fix ---- maybe not anymore --- but frankly: yeah, it does -- aaaa-and maybe fixed now
{

	__shared__ int boundary[6]; // north,east,west,south
	__shared__ int subgridValue;
	__shared__ int newSubgridValue;


	int gid =  blockDim.x*blockIdx.x + threadIdx.x +
			  (blockDim.y*blockIdx.y + threadIdx.y)*blockDim.x*gridDim.x +
			  (blockDim.z*blockIdx.z + threadIdx.z)*blockDim.x*gridDim.x*blockDim.y*gridDim.y;
	double u = cudaSolver->work_u_cuda[gid];
	double u_cmp;
	int subgridValue_cmp=INT_MAX;
	int boundary_index=0;


	if(threadIdx.x+threadIdx.y+threadIdx.z == 0)
	{
		subgridValue = cudaSolver->getSubgridValueCUDA3D(blockIdx.y*gridDim.x + blockIdx.x + blockIdx.z*gridDim.x*gridDim.y);
		boundary[0] = 0;
		boundary[1] = 0;
		boundary[2] = 0;
		boundary[3] = 0;
		boundary[4] = 0;
		boundary[5] = 0;
		newSubgridValue = 0;
//		printf("aaa z = %d, y = %d, x = %d\n",blockIdx.z,blockIdx.y,blockIdx.x);
	}
	__syncthreads();



	if(		(threadIdx.x == 0 				/*				&& !(cudaSolver->currentStep & 1)*/) 		||
			(threadIdx.y == 0 				 	/*			&& (cudaSolver->currentStep & 1)*/) 		||
			(threadIdx.z == 0 	 /*	&& !(cudaSolver->currentStep & 1)*/) 		||
			(threadIdx.x == blockDim.x - 1 	 /*	&& !(cudaSolver->currentStep & 1)*/) 		||
			(threadIdx.y == blockDim.y - 1 	 /*	&& (cudaSolver->currentStep & 1)*/) 		||
			(threadIdx.z == blockDim.z - 1 	 /*	&& (cudaSolver->currentStep & 1)*/) 		)
	{
		if(threadIdx.x == 0 && (blockIdx.x != 0)/* && !(cudaSolver->currentStep & 1)*/)
		{
			u_cmp = cudaSolver->work_u_cuda[gid - 1];
			subgridValue_cmp = cudaSolver->getSubgridValueCUDA3D(blockIdx.y*gridDim.x + blockIdx.x + blockIdx.z*gridDim.x*gridDim.y - 1);
			boundary_index = 2;
		}

		if(threadIdx.x == blockDim.x - 1 && (blockIdx.x != gridDim.x - 1)/* && !(cudaSolver->currentStep & 1)*/)
		{
			u_cmp = cudaSolver->work_u_cuda[gid + 1];
			subgridValue_cmp = cudaSolver->getSubgridValueCUDA3D(blockIdx.y*gridDim.x + blockIdx.x + blockIdx.z*gridDim.x*gridDim.y + 1);
			boundary_index = 1;
		}

		__threadfence();
		if((subgridValue == INT_MAX || fabs(u_cmp) + cudaSolver->delta < fabs(u) ) && (subgridValue_cmp != INT_MAX && subgridValue_cmp != -INT_MAX))
		{
			cudaSolver->unusedCell_cuda[gid] = 0;
			atomicMax(&newSubgridValue, INT_MAX);
			atomicMax(&boundary[boundary_index], 1);
			cudaSolver->work_u_cuda[gid] = u_cmp;
			u=u_cmp;
		}
		__threadfence();
		if(threadIdx.y == 0 && (blockIdx.y != 0)/* && (cudaSolver->currentStep & 1)*/)
		{
			u_cmp = cudaSolver->work_u_cuda[gid - blockDim.x*gridDim.x];
			subgridValue_cmp = cudaSolver->getSubgridValueCUDA3D((blockIdx.y - 1)*gridDim.x + blockIdx.x + blockIdx.z*gridDim.x*gridDim.y);
			boundary_index = 3;
		}
		if(threadIdx.y == blockDim.y - 1 && (blockIdx.y != gridDim.y - 1)/* && (cudaSolver->currentStep & 1)*/)
		{
			u_cmp = cudaSolver->work_u_cuda[gid + blockDim.x*gridDim.x];
			subgridValue_cmp = cudaSolver->getSubgridValueCUDA3D((blockIdx.y + 1)*gridDim.x + blockIdx.x + blockIdx.z*gridDim.x*gridDim.y);
			boundary_index = 0;
		}

		__threadfence();
		if((subgridValue == INT_MAX || fabs(u_cmp) + cudaSolver->delta < fabs(u) ) && (subgridValue_cmp != INT_MAX && subgridValue_cmp != -INT_MAX))
		{
			cudaSolver->unusedCell_cuda[gid] = 0;
			atomicMax(&newSubgridValue, INT_MAX);
			atomicMax(&boundary[boundary_index], 1);
			cudaSolver->work_u_cuda[gid] = u_cmp;
			u=u_cmp;
		}
		__threadfence();

		if(threadIdx.z == 0 && (blockIdx.z != 0)/* && (cudaSolver->currentStep & 1)*/)
		{
			u_cmp = cudaSolver->work_u_cuda[gid - blockDim.x*gridDim.x*blockDim.y*gridDim.y];
			subgridValue_cmp = cudaSolver->getSubgridValueCUDA3D(blockIdx.y*gridDim.x + blockIdx.x + (blockIdx.z - 1)*gridDim.x*gridDim.y);
			boundary_index = 5;
		}
		if(threadIdx.z == blockDim.z - 1 && (blockIdx.z != gridDim.z - 1)/* && (cudaSolver->currentStep & 1)*/)
		{
			u_cmp = cudaSolver->work_u_cuda[gid + blockDim.x*gridDim.x*blockDim.y*gridDim.y];
			subgridValue_cmp = cudaSolver->getSubgridValueCUDA3D(blockIdx.y*gridDim.x + blockIdx.x + (blockIdx.z + 1)*gridDim.x*gridDim.y);
			boundary_index = 4;
		}
		if((subgridValue == INT_MAX || fabs(u_cmp) + cudaSolver->delta < fabs(u) ) && (subgridValue_cmp != INT_MAX && subgridValue_cmp != -INT_MAX))
		{
			cudaSolver->unusedCell_cuda[gid] = 0;
			atomicMax(&newSubgridValue, INT_MAX);
			atomicMax(&boundary[boundary_index], 1);
			cudaSolver->work_u_cuda[gid] = u_cmp;
		}
		__threadfence();

	}
	__syncthreads();

	if(threadIdx.x+threadIdx.y+threadIdx.z == 0)
	{

		if(subgridValue == INT_MAX && newSubgridValue !=0)
			cudaSolver->setSubgridValueCUDA3D(blockIdx.y*gridDim.x + blockIdx.x + blockIdx.z*gridDim.x*gridDim.y, -INT_MAX);

		cudaSolver->setBoundaryConditionCUDA3D(blockIdx.y*gridDim.x + blockIdx.x + blockIdx.z*gridDim.x*gridDim.y, 	1  * boundary[0] +
																													2  * boundary[1] +
																													4  * boundary[2] +
																													8  * boundary[3] +
																													16 * boundary[4] +
																													32 * boundary[5] );
		if(blockIdx.x+blockIdx.y+blockIdx.z == 0)
		{
			cudaSolver->currentStep = cudaSolver->currentStep + 1;
			*(cudaSolver->runcuda) = 0;
		}
	}
}



template <typename SchemeHost, typename SchemeDevice, typename Device>
__global__
void synchronize2CUDA3D(tnlParallelEikonalSolver<3,SchemeHost, SchemeDevice, Device, double, int >* cudaSolver)
{
	int stepValue = cudaSolver->currentStep + 4;
	if( cudaSolver->getSubgridValueCUDA3D(blockIdx.z*gridDim.x*gridDim.y + blockIdx.y*gridDim.x + blockIdx.x) == -INT_MAX )
			cudaSolver->setSubgridValueCUDA3D(blockIdx.z*gridDim.x*gridDim.y + blockIdx.y*gridDim.x + blockIdx.x, stepValue);

	atomicMax((cudaSolver->runcuda),cudaSolver->getBoundaryConditionCUDA3D(blockIdx.z*gridDim.x*gridDim.y + blockIdx.y*gridDim.x + blockIdx.x));
}








template< typename SchemeHost, typename SchemeDevice, typename Device>
__global__
void initCUDA3D( tnlParallelEikonalSolver<3,SchemeHost, SchemeDevice, Device, double, int >* cudaSolver, double* ptr , int* ptr2, int* ptr3)
{


	cudaSolver->work_u_cuda = ptr;//(double*)malloc(cudaSolver->gridCols*cudaSolver->gridRows*cudaSolver->n*cudaSolver->n*sizeof(double));
	cudaSolver->unusedCell_cuda = ptr3;//(int*)malloc(cudaSolver->gridCols*cudaSolver->gridRows*cudaSolver->n*cudaSolver->n*sizeof(int));
	cudaSolver->subgridValues_cuda =(int*)malloc(cudaSolver->gridCols*cudaSolver->gridRows*cudaSolver->gridLevels*sizeof(int));
	cudaSolver->boundaryConditions_cuda =(int*)malloc(cudaSolver->gridCols*cudaSolver->gridRows*cudaSolver->gridLevels*sizeof(int));
	cudaSolver->runcuda = ptr2;//(bool*)malloc(sizeof(bool));
	*(cudaSolver->runcuda) = 1;
	cudaSolver->currentStep = 1;
	//cudaMemcpy(ptr,&(cudaSolver->work_u_cuda), sizeof(double*),cudaMemcpyDeviceToHost);
	//ptr = cudaSolver->work_u_cuda;
	printf("GPU memory allocated.\n");

	for(int i = 0; i < cudaSolver->gridCols*cudaSolver->gridRows*cudaSolver->gridLevels; i++)
	{
		cudaSolver->subgridValues_cuda[i] = INT_MAX;
		cudaSolver->boundaryConditions_cuda[i] = 0;
	}

	/*for(long int j = 0; j < cudaSolver->n*cudaSolver->n*cudaSolver->gridCols*cudaSolver->gridRows; j++)
	{
		printf("%d\n",j);
		cudaSolver->unusedCell_cuda[ j] = 1;
	}*/
	printf("GPU memory initialized.\n");
}




//extern __shared__ double array[];
template< typename SchemeHost, typename SchemeDevice, typename Device >
__global__
void initRunCUDA3D(tnlParallelEikonalSolver<3,SchemeHost, SchemeDevice, Device, double, int >* caller)

{


	extern __shared__ double u[];

	int i =  blockIdx.z *  gridDim.x *  gridDim.y +  blockIdx.y *  gridDim.x +  blockIdx.x;
	int l = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

	__shared__ int containsCurve;
	if(l == 0)
	{
//		printf("z = %d, y = %d, x = %d\n",blockIdx.z,blockIdx.y,blockIdx.x);
		containsCurve = 0;
	}

	caller->getSubgridCUDA3D(i,caller, &u[l]);
	__syncthreads();
	if(u[0] * u[l] <= 0.0)
	{
		atomicMax( &containsCurve, 1);
	}

	__syncthreads();
	if(containsCurve == 1)
	{
		caller->runSubgridCUDA3D(0,u,i);
		__syncthreads();
//		caller->insertSubgridCUDA3D(u[l],i);
		caller->updateSubgridCUDA3D(i,caller, &u[l]);

		__syncthreads();
		if(l == 0)
			caller->setSubgridValueCUDA3D(i, 4);
	}


}





template< typename SchemeHost, typename SchemeDevice, typename Device >
__global__
void runCUDA3D(tnlParallelEikonalSolver<3,SchemeHost, SchemeDevice, Device, double, int >* caller)
{
	extern __shared__ double u[];
	int i =  blockIdx.z *  gridDim.x *  gridDim.y +  blockIdx.y *  gridDim.x +  blockIdx.x;
	int l = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	int bound = caller->getBoundaryConditionCUDA3D(i);

	if(caller->getSubgridValueCUDA3D(i) != INT_MAX && bound != 0 && caller->getSubgridValueCUDA3D(i) > 0)
	{
		caller->getSubgridCUDA3D(i,caller, &u[l]);

		//if(l == 0)
			//printf("i = %d, bound = %d\n",i,caller->getSubgridValueCUDA3D(i));
		if(caller->getSubgridValueCUDA3D(i) == caller->currentStep+4)
		{
			if(bound & 1)
			{
				caller->runSubgridCUDA3D(1,u,i);
				caller->updateSubgridCUDA3D(i,caller, &u[l]);
				__syncthreads();
			}
			if(bound & 2 )
			{
				caller->runSubgridCUDA3D(2,u,i);
				caller->updateSubgridCUDA3D(i,caller, &u[l]);
				__syncthreads();
			}
			if(bound & 4)
			{
				caller->runSubgridCUDA3D(4,u,i);
				caller->updateSubgridCUDA3D(i,caller, &u[l]);
				__syncthreads();
			}
			if(bound & 8)
			{
				caller->runSubgridCUDA3D(8,u,i);
				caller->updateSubgridCUDA3D(i,caller, &u[l]);
				__syncthreads();
			}
			if(bound & 16)
			{
				caller->runSubgridCUDA3D(16,u,i);
				caller->updateSubgridCUDA3D(i,caller, &u[l]);
				__syncthreads();
			}
			if(bound & 32)
			{
				caller->runSubgridCUDA3D(32,u,i);
				caller->updateSubgridCUDA3D(i,caller, &u[l]);
				__syncthreads();
			}

		}
		else
		{
			if( ((bound == 2)))
			{
				caller->runSubgridCUDA3D(2,u,i);
				caller->updateSubgridCUDA3D(i,caller, &u[l]);
				__syncthreads();
			}
			if( ((bound == 1) ))
			{
				caller->runSubgridCUDA3D(1,u,i);
				caller->updateSubgridCUDA3D(i,caller, &u[l]);
				__syncthreads();
			}
			if( ((bound == 8) ))
			{
				caller->runSubgridCUDA3D(8,u,i);
				caller->updateSubgridCUDA3D(i,caller, &u[l]);
				__syncthreads();
			}
			if((bound == 4))
			{
				caller->runSubgridCUDA3D(4,u,i);
				caller->updateSubgridCUDA3D(i,caller, &u[l]);
				__syncthreads();
			}
			if(bound == 16)
			{
				caller->runSubgridCUDA3D(16,u,i);
				caller->updateSubgridCUDA3D(i,caller, &u[l]);
				__syncthreads();
			}
			if(bound == 32)
			{
				caller->runSubgridCUDA3D(32,u,i);
				caller->updateSubgridCUDA3D(i,caller, &u[l]);
				__syncthreads();
			}
		}
																/*  1  2  4  8  16  32  */

		if( ((bound & 19 )))									/*  1  1  0  0   1   0  */
		{
			caller->runSubgridCUDA3D(19,u,i);
			caller->updateSubgridCUDA3D(i,caller, &u[l]);
			__syncthreads();
		}
		if( ((bound & 21 )))									/*  1  0  1  0   1   0  */
		{
			caller->runSubgridCUDA3D(21,u,i);
			caller->updateSubgridCUDA3D(i,caller, &u[l]);
			__syncthreads();
		}
		if( ((bound & 26 )))									/*  0  1  0  1   1   0  */
		{
			caller->runSubgridCUDA3D(26,u,i);
			caller->updateSubgridCUDA3D(i,caller, &u[l]);
			__syncthreads();
		}
		if(   (bound & 28 ))									/*  0  0  1  1   1   0  */
		{
			caller->runSubgridCUDA3D(28,u,i);
			caller->updateSubgridCUDA3D(i,caller, &u[l]);
			__syncthreads();
		}



		if( ((bound & 35 )))									/*  1  0  1  0   0   1  */
		{
			caller->runSubgridCUDA3D(35,u,i);
			caller->updateSubgridCUDA3D(i,caller, &u[l]);
			__syncthreads();
		}
		if( ((bound & 37 )))									/*  1  0  1  0   0   1  */
		{
			caller->runSubgridCUDA3D(37,u,i);
			caller->updateSubgridCUDA3D(i,caller, &u[l]);
			__syncthreads();
		}
		if( ((bound & 42 )))									/*  0  1  0  1   0   1  */
		{
			caller->runSubgridCUDA3D(42,u,i);
			caller->updateSubgridCUDA3D(i,caller, &u[l]);
			__syncthreads();
		}
		if(   (bound & 44 ))									/*  0  0  1  1   0   1  */
		{
			caller->runSubgridCUDA3D(44,u,i);
			caller->updateSubgridCUDA3D(i,caller, &u[l]);
			__syncthreads();
		}

		if(l==0)
		{
			caller->setBoundaryConditionCUDA3D(i, 0);
			caller->setSubgridValueCUDA3D(i, caller->getSubgridValueCUDA3D(i) - 1 );
		}


	}



}

#endif /*HAVE_CUDA*/

#endif /* TNLPARALLELEIKONALSOLVER3D_IMPL_H_ */
