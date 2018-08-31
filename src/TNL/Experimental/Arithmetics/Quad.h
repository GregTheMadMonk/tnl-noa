/*
 *  Quad.h
 *  Quad-C
 *
 *  Created by Matěj Novotný on 27.10.10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

// Secte dva double
// Parametry: a,b
// Vysledek: s
// Chyba: e
// Predpokladame: abs(a) >= abs(b)
void quickTwoSum(double a, double b, double *s, double *e);


// Secte dva double
// Parametry: a,b
// Vysledek: s
// Chyba: e
void twoSum(double a, double b, double *s, double *e);

// Rozdeli double na 2 casti po 26 bitech
// Parametr: a
// Rozdelene cislo: a_hi, a_lo
void split(double a, double *a_hi, double *a_lo);

// Vynasobi dva double
// Parametry: a, b
// Vysledek: p
// Chyba: e
void twoProd(double a, double b, double *p, double *e);

// Normalizace cisla a
// Vztup: pole peti prvku a
// Vystup: pole ctyr prvku b
void renormalize(double *a, double *b);

// Secte double a quad-double
// Parametry: double b, pole ctyr prvku a
// Vysledek: pole ctyr prvku s
void doublePlusQuad(double b, const double *a, double *s);

// Vynasobi double a quad-double
// Parametry: double b, pole ctyr prvku a
// Vysledek: pole ctyr prvku s
void doubleTimesQuad(double b, const double *a, double *s);

void quadDivDouble(const double *a, double b, double *s);

// Secte dva quad-double
// Parametry: pole ctyr prvku a,b
// Vysledek: pole ctyr prvku s
void quadAdd(const double *a, const double *b, double *s);

// Secte dva quad-double, pomalejsi algoritmus, ktery bude v krajnich priadech prejsi
// Parametry: pole ctyr prvku a,b
// Vysledek: pole ctyr prvku s
void quadAddAccurate(const double *a, const double *b, double *s);

// Vynasobi dva quad-double
// Parametry: pole ctyr prvku a,b
// Vysledek: pole ctyr prvku s
void quadMul(const double *a, const double *b, double *s);

// Vynasobi dva quad-double, rychle
// Parametry: pole ctyr prvku a,b
// Vysledek: pole ctyr prvku s
void quadMulQuick(const double *a, const double *b, double *s);

// Vydeli dva quad-double (a / b)
// Parametry: pole ctyr prvku a,b
// Vysledek: pole ctyr prvku s
void quadDiv(const double *a, const double *b, double *s);

// Vynuluje quad-double
// Parametr: pole ctyr prvku a
void zeroQuad(double *a);

// Vytisne quad-double
// Parametr: pole ctyr prvku a
void printQuad(double *a);
                                                          