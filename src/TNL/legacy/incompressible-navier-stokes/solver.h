#include <chrono>

#if 1
 #include "base.h"
#else
 #include "lin_alg.h"
#endif
#include <assert.h>

/*Vytvoreno dle:
http://www.leb.eei.uni-erlangen.de/winterakademie/2008/report/content/course01/pdf/0105.pdf
http://math.mit.edu/~gs/cse/codes/mit18086_navierstokes.pdf

*/

__cuda_call__ inline double HorAvg (const ArrayBaseD & arr, int x, int y) { return 0.5*(arr(x,y) + arr(x+1,y)); }
__cuda_call__ inline double VerAvg (const ArrayBaseD & arr, int x, int y) { return 0.5*(arr(x,y) + arr(x,y+1)); }
__cuda_call__ inline double HorDiff(const ArrayBaseD & arr, int x, int y) { return arr(x+1,y) - arr(x,y); }
__cuda_call__ inline double VerDiff(const ArrayBaseD & arr, int x, int y) { return arr(x,y+1) - arr(x,y); }
__cuda_call__ inline bool IsBoundary(int x, int y, int w, int h){return x==0||y==0||x==w-1||y==h-1;}
__cuda_call__ inline bool IsBoundary(const ArrayBaseD & arr, int x, int y){ return x==0 || y==0 || x==arr.width()-1 || y==arr.height()-1;}
__cuda_call__ inline bool IsOut(const ArrayBaseD & arr, int x, int y){ return x < 0 || y < 0 || x >= arr.width() || y >= arr.height(); }

class RegularMesh
{
	int N;
	bool centerIsBoundary(int x, int y){ return x == 0 || y == 0 || x == N - 1 || y == N - 1; }
	bool verFaceIsBoundary(int x, int y){ return x == 0 || y == 0 || x == N || y == N - 1; }
	bool horFaceIsBoundary(int x, int y){ return x == 0 || y == 0 || x == N -1 || y == N; }
	int center(int x, int y){return y*N + x;}
	int verFace(int x, int y){ return y*(N + 1) + x; }
	int horFace(int x, int y){ return y*N + x; }
	int numCenters() { return N*N; }
	int numVerFaces() { return (N + 1)*N; }
	int numHorFaces() { return N*(N + 1); }

	vec2i verFaceLeftToCenter(int x, int y){ assert(x >= 0);  return vec2i(x, y); }
	vec2i verFaceRightToCenter(int x, int y){ assert(x <  N);  return vec2i(x + 1, y); }
	vec2i horFaceUpToCenter(int x, int y){ assert(y >= 0);  return vec2i(x, y); }
	vec2i horFaceDownToCenter(int x, int y){ assert(y <  N);  return vec2i(x, y+1); }
};

class EmptyMatrix
{
	__cuda_call__ int num_rows() const {return 0;}
	__cuda_call__ int num_cols() const {return 0;}
	__cuda_call__ int num_in_row(int row) const {return 0;}
	__cuda_call__ void get_el_in_row(int row, int ind_in_row, double & val, int &col) const {val=0; col=-1;}
	__cuda_call__ double get_diag(int row) const {return 0;}
	EmptyMatrix& toKernel() { return *this; }
	const EmptyMatrix& toKernel() const { return *this; }
};

class IdentityMatrix
{
	int _size;
public:
	IdentityMatrix() :_size(0){}
	IdentityMatrix(int size) : _size(size){}
	__cuda_call__ int num_rows() const { return _size; }
	__cuda_call__ int num_cols() const { return _size; }
	__cuda_call__ int num_in_row(int row) const { return 1; }
	__cuda_call__ void get_el_in_row(int row, int ind_in_row, double & val, int &col) const { val = 1; col = row; }
	__cuda_call__ double get_diag(int row) const { return 1; }
	IdentityMatrix& toKernel() { return *this; }
	const IdentityMatrix& toKernel() const { return *this; }
};

class SimpleMatrix2D
{
	ArrayBaseD _var;
	double _diag;
	double _off;
public:
	SimpleMatrix2D(){_diag=1;_off=0;}
	SimpleMatrix2D(Array2D<double>& var, double diag, double off){set(var,diag,off);}
	void set(Array2D<double>& var, double diag, double off){_var.set(var); _diag=diag; _off=off;}
	SimpleMatrix2D& toKernel() { return *this; }
	const SimpleMatrix2D& toKernel() const { return *this; }
	__cuda_call__ int num_rows() const {return _var.size();}
	__cuda_call__ int num_cols() const {return _var.size();}
	__cuda_call__ int num_in_row(int row) const {
		vec2i coord = _var.index2D(row);
		return IsBoundary(_var, coord.x, coord.y)? 1 : 5;
	}
	__cuda_call__ void get_el_in_row(int row, int ind_in_row, double & val, int &col) const {
		vec2i coord = _var.index2D(row);
		//if (IsBoundary(_var, coord.x, coord.y)){ col = row; val = 1; return; }
		int x=coord.x, y=coord.y, w = _var.width();
		{
			if (x==0) {col = row+1; val = 1; return;}
			else if (y==0) {col = row+w; val = 1; return;}
			else if (x==_var.width()-1 ) {col = row-1; val = 1; return;}
			else if (y==_var.height()-1) {col = row-w; val = 1; return;}
		}
		switch(ind_in_row)
		{
		case 0: val = _diag; col = row; break;
		case 1: val = _off;  col = row-1; break;
		case 2: val = _off;  col = row+1; break;
		case 3: val = _off;  col = row-_var.width(); break;
		case 4: val = _off;  col = row+_var.width(); break;
		}
	}
};

class AdvectDiffusionMatrix2D
{
public:
	ArrayBaseD u, v;
	double visc, dt;

	AdvectDiffusionMatrix2D(){ visc = dt = 0; }
	AdvectDiffusionMatrix2D(Array2D<double> &u, Array2D<double> &v, double visc, double dt){set(u,v,visc,dt);}
	void set(Array2D<double> &u, Array2D<double> &v, double visc, double dt)
	{
		this->u.set(u); this->v.set(v); this->visc=visc; this->dt = dt;
	}
	AdvectDiffusionMatrix2D& toKernel() { return *this; }
	const AdvectDiffusionMatrix2D& toKernel() const { return *this; }

	__cuda_call__ int num_rows() const {return u.size()+v.size();}
	__cuda_call__ int num_cols() const {return u.size()+v.size();}
	__cuda_call__ int num_in_row(int row) const {
		const ArrayBaseD *act = row>=u.size()? &v : &u;
		vec2i coord = act->index2D(row - (row>=u.size()? u.size() : 0));
		return IsBoundary(*act, coord.x, coord.y)? 1 : 5;
	}
	__cuda_call__ void get_el_in_row(int row, int ind_in_row, double & val, int &col) const
	{
		const ArrayBaseD *act = row>=u.size()? &v : &u;
		vec2i coord = act->index2D(row - (row>=u.size()? u.size() : 0));
		int x=coord.x, y=coord.y, w = act->width();
		if (IsBoundary(*act,x,y)) {col = row; val = 1; return;}

		const double dx = 1.0/u.height(), dy=dx, vix = dt*visc/(dx*dx), viy=dt*visc/(dy*dy);
		double cxm=0,cym=0,cxp=0,cyp=0;
		if (act==&u)
		{
			cxm = -0.25*HorAvg(u,x-1,y)/dx; cxp = 0.25*HorAvg(u,x,y)/dx;
			cym = -0.25*HorAvg(v,x-1,y)/dy; cyp = 0.25*HorAvg(v,x-1,y+1)/dy;
		}
		else
		{
			cxm = -0.25*VerAvg(u,x,y-1)/dx; cxp = 0.25*VerAvg(u,x+1,y-1)/dx;
			cym = -0.25*VerAvg(v,x,y-1)/dy; cyp = 0.25*VerAvg(v,x,y)/dy;
		}
		switch(ind_in_row)
		{
		case 0: val = 1+dt*(cxm+cxp+cym+cyp)+2*vix+2*viy; col = row; break; //Diagonal element
		case 1: val = dt*cxm-vix; col = row-1; break;
		case 2: val = dt*cxp-vix; col = row+1; break;
		case 3: val = dt*cym-viy; col = row-w; break;
		case 4: val = dt*cyp-viy; col = row+w; break;
		case 10: val = 1+2*dt*(cxm+cxp+cym+cyp); col =row; break; //special number for sum of whole row
		}
	}
	__cuda_call__ double get_val_in_row(int row, int ind_in_row) const{
		double val; int col;
		get_el_in_row(row, ind_in_row, val, col);
		return val;
	}
	__cuda_call__ double get_diag(int row) const
	{
		double val; int col;
		get_el_in_row(row, 0, val, col);
		return val;
	}
};

class AdvectModifPoisson
{
	const ArrayBaseD *u,*v, *p;
	const AdvectDiffusionMatrix2D * adMat;
public:
	AdvectModifPoisson(Array2D<double> *p, const AdvectDiffusionMatrix2D * adMat){
		this->p = p;
		this->adMat = adMat;
		this->u = &(adMat->u);
		this->v = &(adMat->v);
	}
	__cuda_call__ int num_rows() const {return p->width()*p->height();}
	__cuda_call__ int num_cols() const {return p->width()*p->height();}
	__cuda_call__ int num_in_row(int row) const {
		vec2i coord = p->index2D(row - (row>=u->size()? u->size() : 0));
		return IsBoundary(*p, coord.x, coord.y)? 1 : 5;
	}
	__cuda_call__ void get_el_in_row(int row, int ind_in_row, double & val, int &col) const
	{

		vec2i coord = p->index2D(row - (row>=u->size()? u->size() : 0));
		int x=coord.x, y=coord.y, w = p->width();
		{
			if (x==0) {col = row+1; val = 1; return;}
			else if (y==0) {col = row+w; val = 1; return;}
			else if (x==w-1) {col = row-1; val = 1; return;}
			else if (y==p->height()-1) {col = row-w; val = 1; return;}
		}

		const int elemInd = 0;
		switch(ind_in_row)
		{
		case 0:
			val = adMat->get_val_in_row( u->index(x-1,y-1), elemInd) + adMat->get_val_in_row( u->index(x,  y-1), elemInd) +
				  adMat->get_val_in_row( v->index(x-1,y-1), elemInd) + adMat->get_val_in_row( v->index(x-1,y  ), elemInd);
			col = row;
			break;
		case 1: val = -adMat->get_val_in_row( u->index(x-1,y-1), elemInd); col = row-1; break;
		case 2: val = -adMat->get_val_in_row( u->index(x,  y-1), elemInd); col = row+1; break;
		case 3: val = -adMat->get_val_in_row( v->index(x-1,y-1), elemInd); col = row-w; break;
		case 4: val = -adMat->get_val_in_row( v->index(x-1,y  ), elemInd); col = row+w; break;
		}
	}
};

#if USE_CUDA
__global__ void GPU_set_zero_neumann(ArrayBaseD a)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int ex = a.w-1, ey=a.h-1;
  if (i < a.w-1) {a(i,0) = a(i,1); a(i,ey) = a(i,ey-1);}
  if (i < a.h-1) {a(0,i) = a(1,i); a(ex,i) = a(ex-1,i);}
  if (i==0)
  {
	  a(0,0)=a(1,1);
	  a(ex,0)=a(ex-1,1);
	  a(0,ey)=a(1,ey-1);
	  a(ex,ey)=a(ex-1,ey-1);
  }
}

__global__ void GPU_set_bnd(ArrayBaseD a, int type) //type is same as enum vars 0=var_u, 1=var_v ...
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int ex = a.w-1, ey=a.h-1;
  double top = type==0? 0.05 : 0;
  if (i < a.w-1) {a(i,0) = 0; a(i,ey) = top;}
  if (i < a.h-1) {a(0,i) = 0; a(ex,i) = 0;}
  if (i==0)
  {
	  a(0,0) = a(ex,0) = a(0,ey) = a(ex,ey) = 0;
  }
}

template <class MatrixType>
__global__ void GPU_pressure_correction(const int dir, double mult, const ArrayBaseD var, MatrixType mat, int indOff, ArrayBaseD res) //dir is 0 for X, or 1 for Y
{
	int x = blockIdx.x*blockDim.x + threadIdx.x, y = blockIdx.y*blockDim.y + threadIdx.y;
	if (IsBoundary(var, x, y)) return;

	double aux = 0;
	if (dir == 0)
		aux += mult*(var(x+1,y+1) - var(x,y+1));
	else
		aux += mult*(var(x+1,y+1) - var(x+1,y));
	int ind = res.index(x,y);
	if (mat.num_rows() > 0) aux /= mat.get_diag(ind+indOff);
	res[ind] += aux;
}
template <class MatrixType>
void Pressure_correction_GPU(double mult, Array2D<double> &u, Array2D<double> &v, Array2D<double> &p,  MatrixType mat, Array<double> & res)
{
	ArrayBaseD pu, pv;
	pu.set(res.data, u.w, u.h, u.d);
	int uOff = u.size();
	pv.set(&(res.data[uOff]), u.w, u.h, u.d);
	GPU_pressure_correction<MatrixType> <<< gridSize2D(u.w, u.h), blockSize2D >>> (0, mult, pu, mat, 0, res);
	GPU_pressure_correction<MatrixType> <<< gridSize2D(v.w, v.h), blockSize2D >>> (0, mult, pv, mat, uOff, res);
}

__global__ void GPU_calc_divergence(const ArrayBaseD u, const ArrayBaseD v, ArrayBaseD res, int N)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x, y = blockIdx.y*blockDim.y + threadIdx.y;
	if (IsOut(res, x, y)) return;
	if (IsBoundary(res, x, y)) return;
	res(x, y) = -0.5f*(u(x, y - 1) - u(x - 1, y - 1) + v(x - 1, y) - v(x - 1, y - 1)) / N; // -(u_x + v_y)
}

template <class MatrixType>
__global__ void GPU_pressure_correction_u_part(const MatrixType mat, const ArrayBaseD u, const ArrayBaseD p, ArrayBaseD res, int N, int sign)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x, y = blockIdx.y*blockDim.y + threadIdx.y;
	if (IsOut(u, x, y)) return;
	if (IsBoundary(u, x, y)) return;

	int ind = u.index(x, y);
	res[ind] += sign*0.5f*N*(p(x + 1, y + 1) - p(x, y + 1)) / mat.get_diag(ind);
}
template <class MatrixType>
__global__ void GPU_pressure_correction_v_part(const MatrixType mat, const ArrayBaseD v, const ArrayBaseD p, ArrayBaseD res, int N, int sign, int vOff)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x, y = blockIdx.y*blockDim.y + threadIdx.y;
	if (IsOut(v, x, y)) return;
	if (IsBoundary(v, x, y)) return;

	int ind = v.index(x, y) + vOff;
	res[ind] += sign*0.5f*N*(p(x + 1, y + 1) - p(x + 1, y)) / mat.get_diag(ind);

}
#endif //USE_CUDA

void Calc_divergence(const Array2D<double> &u, const Array2D<double> &v, Array2D<double> &res, int N)
{
	assert(u.onGPU() == v.onGPU());
	assert(u.onGPU() == res.onGPU());
#if USE_CUDA
	if (res.onGPU())
	{
		GPU_calc_divergence <<< gridSize2D(res.w, res.h), blockSize2D >>> (u, v, res, N);
	}
	else
#endif
	{
		for (int x = 1; x <= N; x++) for (int y = 1; y <= N; y++) {
			res(x, y) = -0.5f*(u(x, y - 1) - u(x - 1, y - 1) + v(x - 1, y) - v(x - 1, y - 1)) / N; // -(u_x + v_y)
		}
	}
}

template <class MatrixType>
static void pressureCorrectionWithA(const Array2D<double> & u, const Array2D<double> & v, const Array2D<double> & p, Array<double> & arr, int sign,
	const MatrixType & mat)
{
	assert(arr.onGPU() == p.onGPU());
	int N = u.height(), vOff = u.size();
#if USE_CUDA
	if (arr.onGPU())
	{
		GPU_pressure_correction_u_part <<< gridSize2D(u.w, u.h), blockSize2D >>> (mat.toKernel(), u, p, arr, N, sign);
		GPU_pressure_correction_v_part <<< gridSize2D(v.w, v.h), blockSize2D >>> (mat.toKernel(), v, p, arr, N, sign, vOff);
		return;
	}
#endif
	
	for (int x = 1; x< u.width() - 1; x++) for (int y = 1; y< u.height() - 1; y++)
	{
		int ind = u.index(x, y);
		arr[ind] += sign*0.5f*N*(p(x,y)-p(x-1,y))/mat.get_diag(ind);
	}
	for (int x = 1; x< v.width() - 1; x++) for (int y = 1; y< v.height() - 1; y++)
	{
		int ind = v.index(x, y) + vOff;
		arr[ind] += sign*0.5f*N*(p(x,y)-p(x,y-1))/mat.get_diag(ind);
	}
}

class NSSolver
{
public:
	enum vars {var_u, var_v, var_p, var_d};
	int N;
	//double diff, visc;

	Array<double> vels, vels0, aux, b;
	Array2D<double> u, u0, v, v0, p, p0, pd, pd0;
	AdvectDiffusionMatrix2D advectNoMat;
	SimpleMatrix2D poissNoMat;
	MatrixCSR advectMat, poissMat;

	NSSolver()
	{
		N=0;
	}

	void init(int size)
	{
		N = size;
		vels.resize(2*N*(N+1)); //for u,v
		vels0.clone(vels);
		aux.clone(vels);
		b.clone(vels); b.fill(0);

		p.resize2d(N,N); p.fill(0); p0.clone(p);
		pd.clone(p); pd0.clone(pd);

		rebind();
		reset();
		advectMat.clone(advectNoMat);
		poissMat.clone(poissNoMat);
	}

	void rebind()
	{
		u.bind2d(vels, 0, N + 1, N);  v.bind2d(vels, N*(N + 1), N, N + 1);
		u0.bind2d(vels0, 0, N + 1, N); v0.bind2d(vels0, N*(N + 1), N, N + 1);
		advectNoMat.set(u, v, 0, 0);
		poissNoMat.set(p, 4, -1);
	}

	void reset()
	{
		u.fill(0); u0.fill(0); v.fill(0); v0.fill(0); p.fill(0); p0.fill(0);
		set_bnd(var_u, u); set_bnd(var_u, u0);
		set_bnd(var_v, v); set_bnd(var_v, v0);
		set_bnd(var_p, p); set_bnd(var_p, p0);
	}

	static void set_zero_neumann(arr2D & a)
	{
		int ex = a.width()-1, ey=a.height()-1;
		for (int x=1; x < ex; x++) {a(x,0) = a(x,1); a(x,ey) = a(x,ey-1);}
		for (int y=1; y < ey; y++) {a(0,y) = a(1,y); a(ex,y) = a(ex-1,y);}
		a(0,0)=a(1,1);
		a(ex,0)=a(ex-1,1);
		a(0,ey)=a(1,ey-1);
		a(ex,ey)=a(ex-1,ey-1);
	}

	static void set_bnd ( vars b, arr2D & x )
	{
#if USE_CUDA
		if (x.onGPU())
		{
			int gs = gridSize(max(x.width(), x.height()));
			if (b==var_p)
				GPU_set_zero_neumann <<< gs, blockSize >>> (x);
			else
				GPU_set_bnd <<< gs, blockSize >>> (x, (int) b);
		}
		else
#endif
		{
			if (b==var_p) {set_zero_neumann(x); return;}
			x.fillBorders(0);
			if (b==var_u) for (int i=1 ; i<x.width()-1 ; i++ ) {x(i,x.height()-1) = 0.05;}
		}
	}

	template <class MatrixType>
	void calcPressure(Array2D<double> & u, Array2D<double> & v, Array2D<double> & p, Array2D<double> & p0, const MatrixType &adMat)
	{
		p0.fill(0); p.fill(0);
		int N = u.height();
		Calc_divergence(u, v, p0, N);
		set_bnd ( var_p, p0 );  //zero neumann bnd. cond.

		for (int i = 0; i < 20; i++)
		{
			JacobiIter(poissNoMat,p0,p,aux); JacobiIter(poissNoMat,p0,aux,p);
			set_bnd ( var_p, p );
		}
		CudaCheckError();

		/*for (int i = 0; i < p.size(); i++)
		{
			double val = p[i];
			if (val != p[i])
				throw "Error: There is a NaN in pressure.";
		}*/
	}
	/*static void pressureCorrection(const Array2D<double> & u, const Array2D<double> & v, const Array2D<double> & p, Array<double> & arr)
	{
		int N = u.height(), vOff = u.size();
		for ( int x=1 ; x< u.width()-1 ; x++ ) for (int y=1 ; y< u.height()-1 ; y++ )
			arr[u.index(x,y)] -= 0.5f*N*(p(x+1,y+1)-p(x,y+1));
		for ( int x=1 ; x< v.width()-1 ; x++ ) for (int y=1 ; y< v.height()-1 ; y++ )
			arr[v.index(x,y)+vOff] -= 0.5f*N*(p(x+1,y+1)-p(x+1,y));
	}*/
	
	/*static void pressureCorrectionWithA2(const Array2D<double> & u, const Array2D<double> & v, const Array2D<double> & p, Array<double> & arr, double sign,
								   const AdvectDiffusionMatrix2D mat)
	{
		int N = u.height(), vOff = u.size();
		for ( int x=1 ; x< u.width()-1 ; x++ ) for (int y=1 ; y< u.height()-1 ; y++ )
		{
			int ind = u.index(x,y);
			arr[ind] += sign*0.5f*N*(p(x+1,y+1)-p(x,y+1))/mat.get_val_in_row(ind, 10);
		}
		for ( int x=1 ; x< v.width()-1 ; x++ ) for (int y=1 ; y< v.height()-1 ; y++ )
		{
			int ind = v.index(x,y)+vOff;
			arr[ind] += sign*0.5f*N*(p(x+1,y+1)-p(x+1,y))/mat.get_val_in_row(ind, 10);
		}
	}*/

	static void createRHS(const Array<double> & vels0, const Array2D<double> & u, const Array2D<double> & v, const Array2D<double> & p, Array<double> & b)
	{
		b.copy(vels0);
		pressureCorrectionWithA(u, v, p, b, -1, IdentityMatrix(b.size()));
		//pressureCorrection(u,v,p,b);
	}

	void prepareAdvectMat(double visc, double dt)
	{
		advectNoMat.set(u, v, visc, dt);
		advectMat.copyVals(advectNoMat);
	}

	void solveAdvectMat(int iter, double damping)
	{
		for (int i = 0; i < iter; i++)
		{
			JacobiIter(advectMat, b, vels, aux, damping);
			JacobiIter(advectMat, b, aux, vels, damping);

		}
	}

	void simulate_velocity(double visc, double dt)
	{
		vels0.copy(vels); p0.copy(p);
		createRHS(vels0, u, v, p, b);
		double residuum = 1e10;
		int count=0;
		prepareAdvectMat(visc,dt);
		auto & matToUse = advectMat;

		for (int i = 0; i < 10; i++)
		{

			solveAdvectMat(5, 0.7);
			set_bnd ( var_u, u ); set_bnd ( var_v, v );
			pressureCorrectionWithA(u, v, p, vels, 1, matToUse);
			calcPressure(u, v, p, p0, matToUse);
			pressureCorrectionWithA(u, v, p, vels, -1, matToUse);

			createRHS(vels0, u, v, p, b);
			residuum = Residuum(matToUse, b, vels);
			CudaCheckError();
			count++;
		}
	}
};
