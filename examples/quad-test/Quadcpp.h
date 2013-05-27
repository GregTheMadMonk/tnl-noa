class QuadDouble {
private:
public:
	double data[4];
	QuadDouble();
	explicit QuadDouble(const double&);
   explicit QuadDouble(int);
	//QuadDouble(double);
	QuadDouble(const QuadDouble&);
	
	//QuadDouble(double*);
	double& operator[](int);
	const double& operator[](int) const;
	QuadDouble& operator =(const QuadDouble&);
	QuadDouble& operator +=(const QuadDouble&);
	QuadDouble& operator -=(const QuadDouble&);
	QuadDouble& operator *=(const QuadDouble&);
	QuadDouble& operator /=(const QuadDouble&);
	QuadDouble& operator =(const double&);
	QuadDouble& operator +=(const double&);
	QuadDouble& operator -=(const double&);
	QuadDouble& operator *=(const double&);
	QuadDouble& operator /=(const double&);
	QuadDouble operator +(const QuadDouble&) const;
	QuadDouble operator -(const QuadDouble&) const;
	QuadDouble operator *(const QuadDouble&) const;
	QuadDouble operator /(const QuadDouble&) const;
	QuadDouble operator +(const double&) const;
	QuadDouble operator -(const double&) const;
	QuadDouble operator *(const double&) const;
	QuadDouble operator /(const double&) const;
	bool operator ==(const QuadDouble&) const;
	bool operator !=(const QuadDouble&) const;
	bool operator <(const QuadDouble&) const;
	bool operator >(const QuadDouble&) const;
	bool operator >=(const QuadDouble&) const;
	bool operator <=(const QuadDouble&) const;
	QuadDouble operator +();
	QuadDouble operator -();
	QuadDouble operator +() const;
	QuadDouble operator -() const;
	operator double() const;	
};

QuadDouble operator +(const double&, const QuadDouble&);
QuadDouble operator -(const double&, const QuadDouble&);
QuadDouble operator *(const double&, const QuadDouble&);
QuadDouble operator /(const double&, const QuadDouble&);
/*double& operator =(const QuadDouble&);
double& operator +=(const QuadDouble&);
double& operator -=(const QuadDouble&);
double& operator *=(const QuadDouble&);
double& operator /=(const QuadDouble&);*/

QuadDouble abs(const QuadDouble&);
QuadDouble sqrt(const QuadDouble&);