#pragma once
#include <math.h>
#include <string.h>

#if USE_CUDA
	#include "base_cuda.h"
#else
	void CudaCheckError() {}
	#define __cuda_call__
#endif

typedef unsigned int uint;
template <class T> T abs(const T &x){return (x<0)? -x:x;}
template <class T> T max(T a, T b){return (a>=b)? a:b;}
template <class T> T max(T a, T b, T c){return max(a, max(a,b));}
template <class T> T min(T a, T b){return (a<=b)? a:b;}
template <class T> inline T square(T x){return x*x;}
template<class T> inline double clamp(T x){ return x<0 ? 0 : x>1 ? 1 : x; }
template <class T> inline void clamp(T & val, T min, T max){if (val<min) val=min; if (val>max) val=max;}

struct vec3
{
  double x, y, z;
  vec3(double x_=0, double y_=0, double z_=0){ x=x_; y=y_; z=z_; }
  vec3& operator+=(const vec3 &b) { x += b.x; y += b.y; z += b.z; return *this; }
  vec3& operator-=(const vec3 &b) { x -= b.x; y -= b.y; z -= b.z; return *this; }
  vec3 operator+(const vec3 &b) const { return vec3(x+b.x,y+b.y,z+b.z); }
  vec3 operator-(const vec3 &b) const { return vec3(x-b.x,y-b.y,z-b.z); }
  vec3 operator*(double b) const { return vec3(x*b,y*b,z*b); }
  vec3 mult(const vec3 &b) const { return vec3(x*b.x,y*b.y,z*b.z); }
  double length() const{return sqrt(x*x+y*y+z*z);}
  vec3& normalize(){ return *this = *this * (1/sqrt(x*x+y*y+z*z)); }
  vec3 norm() const { vec3 res(*this); res.normalize(); return res; }
  vec3& clamp(){ ::clamp<double>(x); ::clamp<double>(y); ::clamp<double>(z); return *this; }
  double dot(const vec3 &b) const { return x*b.x+y*b.y+z*b.z; }
  // cross:
  vec3 operator%(const vec3 & b) const {return vec3(y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x);}
  static double Dot(const vec3 & a, const vec3 & b){ return a.dot(b); }
  static vec3 Cross(const vec3 & a, const vec3 & b){return a%b;}
  static vec3 Mult (const vec3 & a, const vec3 & b){return a.mult(b);}
};

struct vec2i
{
	int x,y;
	__cuda_call__ vec2i(){x=y=0;}
	__cuda_call__ vec2i(int x, int y){this->x=x; this->y=y;}
};

template <class T> struct ArrayBase
{
public:

	int w,h,d; //vector dimensions
	T * data;

	__cuda_call__ ArrayBase (){w=h=d=0; data=0;}
	__cuda_call__ ArrayBase (T *data, int w, int h, int d){this->data=data; this->w=w; this->h=h; this->d=d;}
	__cuda_call__ T & operator [] (int i){return data[i];}
	__cuda_call__ const T & operator [] (int i) const {return data[i];}
	__cuda_call__ T & operator() (int x, int y) { return data[y*w+x]; }
	__cuda_call__ const T & operator() (int x, int y) const { return data[y*w+x]; }
	__cuda_call__ int size () const {return w*h*d;}

	__cuda_call__ int width() const {return w;}
	__cuda_call__ int height() const {return h;}
	__cuda_call__ int depth() const {return d;}

	__cuda_call__ int index(int x, int y) const {return y*w+x;}
	__cuda_call__ vec2i index2D(int i) const { int y = i/w; return vec2i(i-y*w,y);}
	operator T* (){return data;}

	void set(const ArrayBase<T>& arr) {set(arr.data, arr.w, arr.h, arr.d);}
	void set(T *data, int w, int h, int d){this->data=data; this->w=w; this->h=h; this->d=d;}
};
typedef ArrayBase<double> ArrayBaseD;

template<typename T>
class Array : public ArrayBase<T>
{
public:

	Array<T>* _bindedFrom; //If case this is only shared array, this points to the parent data array
	bool _onGPU;

	Array(){ _bindedFrom = 0; _onGPU = false; }
	Array(int size, bool onGPU = false){ this->_onGPU = onGPU; _bindedFrom = 0; resize1d(size); }
	Array(const Array<T> & arr){throw "Not supported";}
	ArrayBase<T> toArr() const {return ArrayBase<T>(this->data,this->w,this->h,this->d);}
	bool onGPU()const { return _bindedFrom ? _bindedFrom->onGPU() : _onGPU; }

	void copy(const Array<T> & arr)
	{
		if (this->size() < arr.size()) throw "Array isn't big enough";
#if USE_CUDA
		cudaMemcpyKind copyKind = arr._onGPU? cudaMemcpyDeviceToHost : cudaMemcpyHostToHost;
		if (_onGPU) copyKind = arr._onGPU? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
		cudaMemcpy(this->data, arr.data, arr.size()* sizeof(T), copyKind);
#else
		memcpy(this->data, arr.data,  arr.size()* sizeof(T));
#endif
	}
	void fill (T val)
	{
#if USE_CUDA
		if (_onGPU)
			FillGPU <<<  gridSize(this->size()), blockSize >>> (this->data, val, this->size());
		else
#else
			for (int i = this->size()-1; i >= 0; i--) this->data[i] = val;
#endif
	}
	void bind1d(Array<T> & arr, int offset, int size){ unbindOrFree(); this->set(&arr[offset], size, 1, 1); _bindedFrom = &arr; }
	void unbindOrFree(){ if (!_bindedFrom) free(); this->data = 0; this->w = this->h = this->d = 0; _bindedFrom = 0; }


	static T* Alloc(int size, bool onGPU)
	{
		T * res = 0;
#if USE_CUDA
		if (onGPU)
			cudaMalloc(&res, size*sizeof(T));
		else
#else
			res = (size > 0) ? new T[size] : 0;
#endif
		return res;
	}
	void free()
	{
		if (_bindedFrom) throw "Cant free not own data";
		if (this->data){
#if USE_CUDA
			if (this->_onGPU)
				cudaFree(this->data);
			else
#else
				delete[] this->data;
#endif
		}
		this->data=0; this->w=this->h=this->d=0; _bindedFrom=0;
	}
	void resize(int newSize, bool leaveMore = false)
	{
		if ( (!leaveMore && this->size() != newSize) || this->size() < newSize)
		{
			free();
			this->data = Alloc(newSize, this->_onGPU);
			_bindedFrom = 0;
		}
		this->w=newSize; this->h=this->d=1;
	}
	void resize1d(int size){resize(size); }
	void clone(const Array<T> & arr, bool leaveMore, bool moveToGPU)
	{
		if ( (!leaveMore && arr.size()!=this->size()) || arr.size() > this->size() )
		{
			free();
			this->_onGPU = moveToGPU;
			resize(arr.size());
		}
		else move(moveToGPU);
		this->w=arr.w; this->h=arr.h; this->d=arr.d;
		this->copy(arr);
	}
	void clone(const Array<T> & arr, bool leaveMore = false){clone(arr, leaveMore, arr._onGPU);}
	~Array(){ if (!_bindedFrom) free(); }

	void move(bool toGPU)
	{
#if USE_CUDA
		if (this->_onGPU == toGPU) return;
		T* newData = Alloc(this->size(), toGPU);
		cudaMemcpy(newData, this->data, this->size()* sizeof(T), toGPU? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost);
		if (this->data){
			if (this->_onGPU) cudaFree(this->data);
			else delete[] this->data;
		}
		this->data = newData;
		this->_onGPU = toGPU;
#else
		assert(false); //CPU only version, what do you want to move?
#endif
	}
	void moveToGPU(){move(true);}
	void moveToCPU(){move(false);}

	void print(const char * name = 0)
	{
		Array<T> aux;
		ArrayBase<T> vec = *this;
		if (_onGPU) { aux.clone(*this, false, false); vec = aux; }
		printf("Printing vector %s\n", (name) ? name : "Noname");
		for (int i = 0; i < this->size(); i++)
			printf("%d:%.10f, ", i, (double)vec.data[i]);
		printf("\n\n");
		fflush(stdout);
	}

	/*T norm()
	{
		T res = 0;
		#pragma omp parallel for reduction(+:res) schedule(static)
		for (int i = 0; i < _size; i++) res+=square(_data[i]);
		return sqrt(res);
	}
	

	void add(const Array<T> & vec, T mult)
	{
		Add(*this, vec, 1.0, mult, *this);
	}

	static void Add(const Array<T> & a, const Array<T> & b, const T aMult, const T bMult, Array<T> & res)
	{
		if (a.size()!=b.size() || b.size() > res.size()) throw "Array::Add - array sizes differ";
		#pragma omp parallel for schedule(static)
		for (int i = 0; i < res.size(); i++)
			res[i]=aMult*a[i]+bMult*b[i];
	}
	void Subtract(const Array<T> & a, const Array<T> & b, Array<T> & res)
	{
		if (a.size()!=b.size() || b.size() > res.size()) throw "Array::Subtract - array sizes differ";
		#pragma omp parallel for schedule(static)
		for (int i = 0; i < b.size(); i++)
			res[i]=a[i]-b[i];
	}*/
};

typedef Array<double> ArrayD;
typedef Array<int> ArrayI;

template <class T>
class Array2D: public Array<T>
{
public:
	void resize2d(int width, int height){if (width==this->w && height==this->h) return; this->resize(width*height);
		this->w=width; this->h=height; this->d=1;}
	void bind2d(Array<T> & arr, int offset, int width, int height){
		this->unbindOrFree();
		this->set(&arr.data[offset], width, height, 1); 
		this->_bindedFrom=&arr;}
	void bind3d(Array<T> & arr, int offset, int width, int height, int depth){this->unbindOrFree(); 
		this->set(&arr[offset], width, height, depth); this->_bidnFrom=&arr;}
	void clone(const Array2D<T>& arr)
	{
		Array<T>::clone(arr);
		this->w = arr.width(); this->h=arr.height();
	}
	void fillBorders(T val){
		if (this->_onGPU) throw "FillBorders isn't implemented on GPU yet.";
		for (int x=0; x < this->w;   x++) this->data[x] = this->data[(this->h-1)*this->w + x] = val;
		for (int y=1; y < this->h-1; y++) this->data[y*this->w] = this->data[y*this->w + this->w-1] = val;
	}
};
typedef Array2D<double> arr2D;


struct GPUMatrix
{
	int _num_rows, _num_cols;
	ArrayBaseD _vals;
	ArrayBase<int> _cols, _rowStarts;

	__cuda_call__ int num_rows() const { return _num_rows; }
	__cuda_call__ int num_cols() const { return _num_cols; }
	__cuda_call__ int num_in_row(int row) const { return _rowStarts[row + 1] - _rowStarts[row]; }
	__cuda_call__ inline void get_el_in_row(int row, int ind_in_row, double & out_val, int & out_col) const {
		int i = _rowStarts[row] + ind_in_row; out_val = _vals[i]; out_col = _cols[i];
	}
	inline double& get_val_in_row(int row, int ind_in_row){ return _vals[_rowStarts[row] + ind_in_row]; }
	__cuda_call__ inline double  get_val_in_row(int row, int ind_in_row) const { return _vals[_rowStarts[row] + ind_in_row]; }
	__cuda_call__ inline int get_col_index(int row, int ind_in_row) const { return _cols[_rowStarts[row] + ind_in_row]; }
	__cuda_call__ double get_diag(int row) const
	{
		for (int i = _rowStarts[row]; i < _rowStarts[row + 1]; i++)
			if (_cols[i] == row) return _vals[i];
		return -1e100;
		//throw "Diagonal element not found";
	}
};

class MatrixCSR
{
public:
	ArrayD _vals;
	ArrayI _cols;
	ArrayI _rowStarts;
	int _num_rows, _num_cols;

	MatrixCSR(){_num_cols=_num_rows=0;}
	MatrixCSR(const MatrixCSR & mat){throw "Copy constructor for MatrixCSR doesn't exist";}
	__cuda_call__ int num_rows() const { return _num_rows; }
	__cuda_call__ int num_cols() const { return _num_cols; }
	__cuda_call__ int num_in_row(int row) const { return _rowStarts[row + 1] - _rowStarts[row]; }
	GPUMatrix toKernel() const
	{
		assert(_vals.onGPU());
		GPUMatrix res;
		res._num_rows = _num_rows; res._num_cols = _num_cols;
		res._rowStarts = _rowStarts;
		res._cols = _cols; res._vals = _vals;
		return res;
	}
	void resize(int num_rows, int num_cols, int num_values)
	{
		if (_vals.size()!=num_values)
		{
			_vals.resize(num_values);
			_cols.resize(num_values);
		}
		if (_rowStarts.size()!=num_rows+1) _rowStarts.resize(num_rows+1);
		_rowStarts.fill(0);
		_num_rows=num_rows;
		_num_cols=num_cols;
	}
	void clear()
	{
		_vals.fill(0);
		_cols.fill(0);
		_rowStarts.fill(0);
	}

	template <class MatrixType>
	void copyVals(const MatrixType & matToClone)
	{
		for (int ri = 0; ri < _num_rows; ri++)
		{
			int nr = num_in_row(ri), rs = _rowStarts[ri];
			for (int i = 0; i < nr; i++)
				_vals[rs + i] = matToClone.get_val_in_row(ri, i);
		}
	}

	template <class MatrixType>
	void clone(const MatrixType & matToClone)
	{
		_num_rows = matToClone.num_rows();
		_num_cols = matToClone.num_cols();
		_rowStarts.resize(_num_rows+1);
		_rowStarts[0] = 0;
		for (int ri = 0; ri < _num_rows; ri++)
			_rowStarts[ri+1] = _rowStarts[ri] + matToClone.num_in_row(ri);

		int nne = _rowStarts[_num_rows];
		_vals.resize(nne);
		_cols.resize(nne);
		for (int ri = 0; ri < _num_rows; ri++)
		{
			int nr = matToClone.num_in_row(ri), rs = _rowStarts[ri];
			for (int i = 0; i < nr; i++)
				matToClone.get_el_in_row(ri,i,_vals[rs+i],_cols[rs+i]);
		}
	}

	inline double& operator()(int ri, int ci)
	{
		if(ri>=_num_rows || ci>=_num_cols) throw "MatrixCSR - Index out of bounds";
		for (int i = _rowStarts[ri]; i < _rowStarts[ri+1]; i++ )
			if (_cols[i] == ci) return _vals[i];
		throw "MatrixCSR - Value not found";
	}
	inline const double& operator()(int ri, int ci) const
	{
		if(ri>=_num_rows || ci>=_num_cols) throw "MatrixCSR - Index out of bounds";
		for (int i = _rowStarts[ri]; i < _rowStarts[ri+1]; i++ )
			if (_cols[i] == ci) return _vals[i];
		throw "MatrixCSR - Value not found";
	}
	inline bool isNull(int ri, int ci) const
	{
		if(ri>=_num_rows || ci>=_num_cols) throw "MatrixCSR - Index out of bounds";
		for (int i = _rowStarts[ri]; i < _rowStarts[ri+1]; i++ )
			if (_cols[i] == ci) return false;
		return true;
	}
	__cuda_call__ inline void get_el_in_row(int row, int ind_in_row, double & out_val, int & out_col) const {
		int i = _rowStarts[row] + ind_in_row; out_val = _vals[i]; out_col=_cols[i];
	}
	inline double& get_val_in_row(int row, int ind_in_row){return _vals[_rowStarts[row] + ind_in_row];}
	__cuda_call__ inline double  get_val_in_row(int row, int ind_in_row) const { return _vals[_rowStarts[row] + ind_in_row]; }
	__cuda_call__ inline int get_col_index(int row, int ind_in_row) const { return _cols[_rowStarts[row] + ind_in_row]; }
	__cuda_call__ double get_diag(int row) const
	{ 
		for (int i = _rowStarts[row]; i < _rowStarts[row + 1]; i++)
			if (_cols[i] == row) return _vals[i];
		return -1e100;
		//throw "Diagonal element not found";
	}
	void loadMMMatrix(const char * filename);
	void mult(const double * vec, double *out) const
	{
		#pragma omp parallel for schedule(static)
		for (int ri = 0; ri < _num_rows; ri++)
		{
			out[ri] = 0;
			for (int i = _rowStarts[ri]; i < _rowStarts[ri+1]; i++)
				out[ri] += _vals[i] * vec[_cols[i]];
		}
	}
	void mult(const ArrayD & vec, ArrayD & out) const
	{
		if (out.size()< _num_rows) throw "CSRMatrix::mult - out vector not big enough";
		if (vec.size()< _num_cols) throw "CSRMatrix::mult - vec vector not big enough";
		#pragma omp parallel for schedule(static)
		for (int ri = 0; ri < _num_rows; ri++)
		{
			out[ri] = 0;
			for (int i = _rowStarts[ri]; i < _rowStarts[ri+1]; i++)
				out[ri] += _vals[i] * vec[_cols[i]];
		}
	}

	void print() const
	{
		printf("Matrix %d x %d with %d values\n", _num_rows, _num_cols, _vals.size()); fflush(stdout);
		for (int ri = 0; ri < _num_rows; ri++)
		{
			printf("Row %d - ", ri);
			for (int i = _rowStarts[ri]; i < _rowStarts[ri+1]; i++)
				printf("%d:%f, ",_cols[i], _vals[i]);
			printf("\n");
		}
		fflush(stdout);
	}

	void moveToGPU(){ _rowStarts.moveToGPU(); _cols.moveToGPU(); _vals.moveToGPU(); }
	void moveToCPU(){ _rowStarts.moveToCPU(); _cols.moveToCPU(); _vals.moveToCPU(); }

	/*static void MatrixMult(const MatrixCSR & A, const MatrixCSR & B, MatrixCSR & res)
	{
		if (A._num_cols != B._num_rows) throw "CSRMatrix::MatrixMult - A.cols and b.rows don't agree";

		intArr aux(B.num_cols()); aux.fill(-1);
		res._rowStarts.resize(A.num_rows());
		for (int r = 0; r < A.num_rows(); r++)
		{
			int actCols=0;
			for (int i = 0; i < A.num_in_row(r); i++)
			{
				int ri = A.get_col_index(r,i);
				for (int j = 0; j < B.num_in_row(ri); j++)
				{
					int ci = B.get_col_index(ri,j);
					if (aux[ci]!=r)
					{
						actCols++;
						aux[ci] = r;
					}
				}
			}
			res._rowStarts[r+1] = res._rowStarts[r+1]+actCols;
		}

	#pragma omp parallel for schedule(static)
		for (int r = 0; r < A.num_rows(); r++)
		{
			for (int i = 0; i < A.num_in_row(r); i++)
			{
				int ri = A.get_col_index(r,i);
				for (int j = 0; j < B.num_in_row(ri); j++)
				{
					int ci = B.get_col_index(ri,j);
					double val = res(r,ci);
					res(r,ci) = val + A.get_val_in_row(r,i)*B.get_val_in_row(ri,j);
				}
			}
		}
	}*/
};
