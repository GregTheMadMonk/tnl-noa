#include "Quad.h"
#include "Quadcpp.h"
#include <math.h>


QuadDouble::QuadDouble() {
	zeroQuad(data);
}

QuadDouble::QuadDouble(const double& value) {
	data[0] = value;
	data[1] = 0;
	data[2] = 0;
	data[3] = 0;
}

QuadDouble::QuadDouble(int value) {
	data[0] = (double)value;
	data[1] = 0;
	data[2] = 0;
	data[3] = 0;
}

QuadDouble::QuadDouble(const QuadDouble& other) {
	data[0] = other[0];
	data[1] = other[1];
	data[2] = other[2];
	data[3] = other[3];
}

/*QuadDouble::QuadDouble(double* other) {
    double d[5];
	d[0] = other[0];
	d[1] = other[1];
	d[2] = other[2];
	d[3] = other[3];
    d[4] = 0;
    renormalize(d, data);	
}*/

double& QuadDouble::operator [](int idx) {
	return data[idx];
}

const double& QuadDouble::operator [](int idx) const{
	return data[idx];
}

QuadDouble& QuadDouble::operator =(const QuadDouble& rhs) {
	data[0] = rhs[0];
	data[1] = rhs[1];
	data[2] = rhs[2];
	data[3] = rhs[3];
	return *this;
}

QuadDouble& QuadDouble::operator +=(const QuadDouble& rhs) {
	quadAddAccurate(data, rhs.data, data);
	return *this;
}

QuadDouble& QuadDouble::operator -=(const QuadDouble& rhs) {
	quadAddAccurate(data, (-rhs).data, data);
	return *this;
}

QuadDouble& QuadDouble::operator *=(const QuadDouble& rhs) {
	quadMul(data, rhs.data, data);
	return *this;
}

QuadDouble& QuadDouble::operator /=(const QuadDouble& rhs) {
	quadDiv(data, rhs.data, data);
	return *this;
}

QuadDouble& QuadDouble::operator =(const double& rhs) {
	data[0] = rhs;
	data[1] = 0;
	data[2] = 0;
	data[3] = 0;
	return *this;
}

QuadDouble& QuadDouble::operator +=(const double& rhs) {
	doublePlusQuad(rhs, data, data);
	return *this;
}

QuadDouble& QuadDouble::operator -=(const double& rhs) {
	doublePlusQuad(-rhs, data, data);
	return *this;
}

QuadDouble& QuadDouble::operator *=(const double& rhs) {
	doubleTimesQuad(rhs, data, data);
	return *this;
}

QuadDouble& QuadDouble::operator /=(const double& rhs) {
	quadDivDouble(data, rhs, data);
	return *this;
}

QuadDouble QuadDouble::operator +(const QuadDouble& value) const{
	QuadDouble qd(*this);
	qd += value;
	return qd;
}

QuadDouble QuadDouble::operator -(const QuadDouble& value) const{
	QuadDouble qd(*this);
	qd -= value;
	return qd;
}

QuadDouble QuadDouble::operator *(const QuadDouble& value) const{
	QuadDouble qd(*this);
	qd *= value;
	return qd;
}

QuadDouble QuadDouble::operator /(const QuadDouble& value) const{
	QuadDouble qd(*this);
	qd /= value;
	return qd;
}


QuadDouble QuadDouble::operator +(const double& value) const {
	QuadDouble qd(*this);
	qd += value;
	return qd;
}

QuadDouble QuadDouble::operator -(const double& value) const {
	QuadDouble qd(*this);
	qd -= value;
	return qd;
}

QuadDouble QuadDouble::operator *(const double& value) const {
	QuadDouble qd(*this);
	qd *= value;
	return qd;
}

QuadDouble QuadDouble::operator /(const double& value) const {
	QuadDouble qd(*this);
	qd /= value;
	return qd;
}

bool QuadDouble::operator ==(const QuadDouble& rhs) const {
	if (data[0] == rhs[0] && data[1] == rhs[1] && data[2] == rhs[2] && data[3] == rhs[3]) {
		return true;
	}
	return false;
}

/*bool QuadDouble::operator ==(const QuadDouble& rhs) {
	if (data[0] == rhs[0] && data[1] == rhs[1] && data[2] == rhs[2] && data[3] == rhs[3]) {
		return true;
	}
	return false;
}*/

bool QuadDouble::operator !=(const QuadDouble& rhs) const {
	return !(*this == rhs);
}


bool QuadDouble::operator <(const QuadDouble& rhs) const {
	QuadDouble qd(*this);
	qd -= rhs;
	if (qd[0] < 0.) {
		return true;
	}
	return false;
}

bool QuadDouble::operator >(const QuadDouble& rhs) const {
	QuadDouble qd(*this);
	qd -= rhs;
	if (qd[0] > 0.) {
		return true;
	}
	return false;
}
 
bool QuadDouble::operator >=(const QuadDouble& rhs) const {
	QuadDouble qd(*this);
	qd -= rhs;
	if (qd[0] >= 0.) {
		return true;
	}
	return false;
}

bool QuadDouble::operator <=(const QuadDouble& rhs) const  {
	QuadDouble qd(*this);
	qd -= rhs;
	if (qd[0] <= 0.) {
		return true;
	}
	return false;
}

QuadDouble QuadDouble::operator +() {
	QuadDouble qd(*this);
	return qd;
}

QuadDouble QuadDouble::operator -() {
	QuadDouble qd(*this);
	qd[0] = -qd[0];
	qd[1] = -qd[1];
	qd[2] = -qd[2];
	qd[3] = -qd[3];
	return qd;
}

QuadDouble QuadDouble::operator +() const {
	QuadDouble qd(*this);
	return qd;
}

QuadDouble QuadDouble::operator -() const {
	QuadDouble qd(*this);
	qd[0] = -qd[0];
	qd[1] = -qd[1];
	qd[2] = -qd[2];
	qd[3] = -qd[3];
	return qd;
}

QuadDouble::operator double() const{
	return data[0];
}

QuadDouble operator+(const double& v1, const QuadDouble& v2) {
    QuadDouble qd(v1);
    qd += v2;
    return qd;
}

QuadDouble operator-(const double& v1, const QuadDouble& v2) {
    QuadDouble qd(v1);
    qd -= v2;
    return qd;
}

QuadDouble operator*(const double& v1, const QuadDouble& v2) {
    QuadDouble qd(v1);
    qd *= v2;
    return qd;
}

QuadDouble operator/(const double& v1, const QuadDouble& v2) {
    QuadDouble qd(v1);
    qd /= v2;
    return qd;
}

QuadDouble abs(const QuadDouble& value) {
	QuadDouble qd(value);
	if (value[0] < 0) {
		qd = -qd;
	}
	return qd;
}

QuadDouble sqrt(const QuadDouble& value) {
	QuadDouble qd(value);
	QuadDouble x(1/sqrt((double)qd));
	QuadDouble step;
	//TODO zjednodušit dělení 2
	step = x * (1. - qd * x * x);
	step[0] /= 2;
	step[1] /= 2;
	step[2] /= 2;
	step[3] /= 2;
	x += step;
	step = x * (1. - qd * x * x);
	step[0] /= 2;
	step[1] /= 2;
	step[2] /= 2;
	step[3] /= 2;
	x += step;
	step = x * (1. - qd * x * x);
	step[0] /= 2;
	step[1] /= 2;
	step[2] /= 2;
	step[3] /= 2;
	x += step;
	qd *= x;
	return qd;
}