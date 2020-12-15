/*
 ============================================================================
 Name        : nanoBraggCUDA.cu
 Author      :
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include "nanotypes.h"
#include "cuda_compatibility.h"

using simtbx::nanoBragg::shapetype;
using simtbx::nanoBragg::hklParams;

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

#ifndef CUDAREAL
#define CUDAREAL float
#endif

cudaError_t cudaMemcpyVectorDoubleToDevice(CUDAREAL *dst, double *src, size_t vector_items);

/* make a unit vector pointing in same direction and report magnitude (both args can be same vector) */
double cpu_unitize(double *vector, double *new_unit_vector);

__global__ void nanoBraggSpotsInitCUDAKernel(int spixels, int fpixesl, float * floatimage, float * omega_reduction, float * max_I_x_reduction,
		float * max_I_y_reduction, bool * rangemap);

__global__ void nanoBraggSpotsCUDAKernel(int spixels, int fpixels, int roi_xmin, int roi_xmax, int roi_ymin, int roi_ymax, int oversample, int point_pixel,
CUDAREAL pixel_size, CUDAREAL subpixel_size, int steps, CUDAREAL detector_thickstep, int detector_thicksteps, CUDAREAL detector_thick, CUDAREAL detector_mu,
		const CUDAREAL * __restrict__ sdet_vector, const CUDAREAL * __restrict__ fdet_vector, const CUDAREAL * __restrict__ odet_vector,
		const CUDAREAL * __restrict__ pix0_vector, int curved_detector, CUDAREAL distance, CUDAREAL close_distance, const CUDAREAL * __restrict__ beam_vector,
		CUDAREAL Xbeam, CUDAREAL Ybeam, CUDAREAL dmin, CUDAREAL phi0, CUDAREAL phistep, int phisteps, const CUDAREAL * __restrict__ spindle_vector, int sources,
		const CUDAREAL * __restrict__ source_X, const CUDAREAL * __restrict__ source_Y, const CUDAREAL * __restrict__ source_Z,
		const CUDAREAL * __restrict__ source_I, const CUDAREAL * __restrict__ source_lambda, const CUDAREAL * __restrict__ a0, const CUDAREAL * __restrict__ b0,
		const CUDAREAL * __restrict c0, shapetype xtal_shape, CUDAREAL mosaic_spread, int mosaic_domains, const CUDAREAL * __restrict__ mosaic_umats,
		CUDAREAL Na, CUDAREAL Nb,
		CUDAREAL Nc, CUDAREAL V_cell,
		CUDAREAL water_size, CUDAREAL water_F, CUDAREAL water_MW, CUDAREAL r_e_sqr, CUDAREAL fluence, CUDAREAL Avogadro, CUDAREAL spot_scale, int integral_form, CUDAREAL default_F,
		int interpolate, const CUDAREAL * __restrict__ Fhkl, const hklParams * __restrict__ Fhklparams, int nopolar, const CUDAREAL * __restrict__ polar_vector, CUDAREAL polarization, CUDAREAL fudge,
		const int unsigned short * __restrict__ maskimage, float * floatimage /*out*/, float * omega_reduction/*out*/, float * max_I_x_reduction/*out*/,
		float * max_I_y_reduction /*out*/, bool * rangemap);


extern "C" void nanoBraggSpotsCUDA(int deviceId, int spixels, int fpixels, int roi_xmin, int roi_xmax, int roi_ymin, int roi_ymax, int oversample, int point_pixel,
                double pixel_size, double subpixel_size, int steps, double detector_thickstep, int detector_thicksteps, double detector_thick, double detector_mu,
                double sdet_vector[4], double fdet_vector[4], double odet_vector[4], double pix0_vector[4], int curved_detector, double distance, double close_distance,
                double beam_vector[4], double Xbeam, double Ybeam, double dmin, double phi0, double phistep, int phisteps, double spindle_vector[4], int sources,
                double *source_X, double *source_Y, double * source_Z, double * source_I, double * source_lambda, double a0[4], double b0[4], double c0[4],
                shapetype xtal_shape, double mosaic_spread, int mosaic_domains, double * mosaic_umats, double Na, double Nb, double Nc, double V_cell,
                double water_size, double water_F, double water_MW, double r_e_sqr, double fluence, double Avogadro, int integral_form, double default_F,
                int interpolate, double *** Fhkl, int h_min, int h_max, int h_range, int k_min, int k_max, int k_range, int l_min, int l_max, int l_range, int hkls,
                int nopolar, double polar_vector[4], double polarization, double fudge, int unsigned short * maskimage, float * floatimage /*out*/,
                double * omega_sum/*out*/, int * sumn /*out*/, double * sum /*out*/, double * sumsqr /*out*/, double * max_I/*out*/, double * max_I_x/*out*/,
                double * max_I_y /*out*/, double spot_scale);

/* cubic spline interpolation functions */
__device__ void polint(const CUDAREAL *xa, const CUDAREAL *ya, CUDAREAL x, CUDAREAL *y);
__device__ void polin2(CUDAREAL *x1a, CUDAREAL *x2a, CUDAREAL ya[4][4], CUDAREAL x1, CUDAREAL x2, CUDAREAL *y);
__device__ void polin3(CUDAREAL *x1a, CUDAREAL *x2a, CUDAREAL *x3a, CUDAREAL ya[4][4][4], CUDAREAL x1, CUDAREAL x2, CUDAREAL x3, CUDAREAL *y);
/* rotate a 3-vector about a unit vector axis */
__device__ CUDAREAL *rotate_axis(const CUDAREAL * __restrict__ v, CUDAREAL *newv, const CUDAREAL * __restrict__ axis, const CUDAREAL phi);
__device__ CUDAREAL *rotate_axis_ldg(const CUDAREAL * __restrict__ v, CUDAREAL * newv, const CUDAREAL * __restrict__ axis, const CUDAREAL phi);
/* make a unit vector pointing in same direction and report magnitude (both args can be same vector) */
__device__ CUDAREAL unitize(CUDAREAL * vector, CUDAREAL * new_unit_vector);
/* vector cross product where vector magnitude is 0th element */
__device__ CUDAREAL *cross_product(CUDAREAL * x, CUDAREAL * y, CUDAREAL * z);
/* vector inner product where vector magnitude is 0th element */
__device__ CUDAREAL dot_product(const CUDAREAL * x, const CUDAREAL * y);
__device__ CUDAREAL dot_product_ldg(const CUDAREAL * __restrict__ x, CUDAREAL * y);
/* measure magnitude of vector and put it in 0th element */
__device__ void magnitude(CUDAREAL *vector);
/* scale the magnitude of a vector */
__device__ CUDAREAL vector_scale(CUDAREAL *vector, CUDAREAL *new_vector, CUDAREAL scale);
/* rotate a 3-vector using a 9-element unitary matrix */
__device__ void rotate_umat_ldg(CUDAREAL * v, CUDAREAL *newv, const CUDAREAL * __restrict__ umat);
/* Fourier transform of a truncated lattice */
__device__ CUDAREAL sincg(CUDAREAL x, CUDAREAL N);
__device__ CUDAREAL sincgrad(CUDAREAL x, CUDAREAL N);
/* Fourier transform of a sphere */
__device__ CUDAREAL sinc3(CUDAREAL x);
/* polarization factor from vectors */
__device__ CUDAREAL polarization_factor(CUDAREAL kahn_factor, CUDAREAL *incident, CUDAREAL *diffracted, const CUDAREAL * __restrict__ axis);

__device__ __inline__  int flatten3dindex(int x, int y, int z, int x_range, int y_range, int z_range);

__device__ __inline__ CUDAREAL quickFcell_ldg(int hkls, int h_max, int h_min, int k_max, int k_min, int l_min, int l_max, int h0, int k0, int l0, int h_range, int k_range, int l_range, CUDAREAL defaultF, const CUDAREAL * __restrict__ Fhkl);
