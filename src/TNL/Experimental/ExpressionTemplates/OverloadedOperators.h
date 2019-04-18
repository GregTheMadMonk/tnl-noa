/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   OverloadedOperators.h
 * Author: legler
 *
 * Created on December 4, 2018, 7:55 PM
 */

#ifndef OVERLOADEDOPERATORS_H
#define OVERLOADEDOPERATORS_H

#include <vector>

//using std::vector;

template< class T >
std::vector<T> operator+(const std::vector<T> &a, const std::vector<T> &b)
{
std::vector<T> res(a.size());
for (std::size_t i = 0; i<a.size(); i++)
res[i] = a[i] + b[i];
return res;
}

template< class T >
std::vector<T> operator-(const std::vector<T> &a, const std::vector<T> &b)
{
std::vector<T> res(a.size());
for (std::size_t i = 0; i<a.size(); i++)
res[i] = a[i] + b[i];
return res;
}
/*
template< class T>
std::vector<T> operator*(const T &a, const std::vector<T> &b)
{
std::vector<T> res(a.size());
for (size_t i = 0; i<a.size(); i++)
res[i] = a + b[i];
return res;
}

template< class T>
std::vector<T> operator*(const std::vector<T> &a, const T &b)
{
std::vector<T> res(a.size());
for (size_t i = 0; i<a.size(); i++)
res[i] = a[i] + b;
return res;
}
*/
/*
template<typename T>
class Vec
{
public:
	std::vector<T> data;
	
	Vec(const std::size_t size) : data(size)
	{}
	
	Vec(const std::size_t size, const double init) : data(size, init)
	{}

        Vec operator+ ( const Vec& a,  const Vec& b )
        {
            Vec<T> res(a.data.size());
            for (std::size_t i = 0; i<a.data.size(); i++)
            res[i] = a[i] + b[i];
            return res;
        }
        
        Vec operator- ( const Vec& a,  const Vec& b )
        {
            Vec<T> res(a.data.size());
            for (std::size_t i = 0; i<a.data.size(); i++)
            res[i] = a[i] - b[i];
            return res;
        }
        
        Vec operator* ( const T& a,  const Vec& b )
        {
            Vec<T> res(b.data.size());
            for (std::size_t i = 0; i<b.data.size(); i++)
            res[i] = a * b[i];
            return res;
        }
};
*/

#endif /* OVERLOADEDOPERATORS_H */

