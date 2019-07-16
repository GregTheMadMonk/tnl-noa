\page tutorial_01_arrays Flexible (parallel) reduction tutorial

## Introduction

This tutorial introduces flexible parallel reduction in TNL. It shows how to easily implement parallel reduction with user defined operations which may run on both CPU and GPU. Parallel reduction is a programming pattern appering very often in different kind of algorithms for example in scalar product, vector norms or mean value evaluation but also in sequences or strings comparison.

## Table of Contents
1. [Flexible Parallel Reduction](#flexible_parallel_reduction)
   1. [Sum](#flexible_parallel_reduction_sum)

## Flexible parallel reduction<a name="flexible_parallel_reduction"></a>

We will explain the *flexible parallel reduction* on several examples. We start with the simplest sum of sequence of numbers folowed by more advanced problems like scalar product or vector norms.

### Sum

We start with simple problem.

