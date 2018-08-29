#include <stdio.h>
#include <math.h>
#include <time.h>

#include "diffusion.h"

#define PI 4.0*atan(1.0)

// Solve the 1D diffusion equation with an uncertain variable
// coefficient using finite differences and TDMA.
//
// del(k del(u) ) = f on [0,1] subject to u(0) = 0 u(1) = 0
//
// Here we set f = -10 and k is a random variable

float diffusion_1d(size_t num_grid_points, size_t num_uncertainties, float *xi_uncertainties) {

  // Problem parameters
  float domain_length = 1.0;		// Length of the domain (starts at 0)
  float u_0 = 0.0;			// Set left boundary value for unknowns
  float u_1 = 0.0;			// Set right boundary value for unknowns
  float f = -10.0;			// Set forcing term value
  float sigma = 1.0;			// Set variability of diffusivity

  // Generate grid array
  float x_points[num_grid_points];	// Includes boundary points
  for(size_t i = 0; i < num_grid_points; i++) {

    // Grid spacing
    float grid_spacing = domain_length/(num_grid_points-1);

    // Set points
    x_points[i] = i*grid_spacing;

  }

  // Generate & set forcing term array
  float f_terms[num_grid_points];	// Includes boundary points
  for(size_t i = 0; i < num_grid_points; i++) {
    // Initialize
    f_terms[i] = f;
  }

  // Generate & compute stochastic coefficient array
  float k_coefficients[num_grid_points];	// Includes boundary points
  for(size_t i = 0; i < num_grid_points; i++) {

    // Initialize
    k_coefficients[i] = 1.0;
    for(size_t k = 0; k < num_uncertainties; k++) {
      k_coefficients[i] += sigma*((1.0/((k+1.0)*(k+1.0)*PI*PI))*cos(2.0*PI*(k+1.0)*x_points[i])*xi_uncertainties[k]);
    }
  }

  // Solve linear system of equations using TDMA (boundary points solved using identity)
  float a[num_grid_points];	// Includes boundary points
  float b[num_grid_points];	// Includes boundary points
  float c[num_grid_points];	// Includes boundary points
  float d[num_grid_points];	// Includes boundary points
  for(size_t i = 1; i < (num_grid_points-1); i++) {

    a[i] = 0.5*(k_coefficients[i]+k_coefficients[i-1])/(x_points[i]-x_points[i-1]);
    b[i] = (-1.0)*0.5*(k_coefficients[i+1]+k_coefficients[i])/(x_points[i+1]-x_points[i])+(-1.0)*0.5*(k_coefficients[i]+k_coefficients[i-1])/(x_points[i]-x_points[i-1]);
    c[i] = 0.5*(k_coefficients[i+1]+k_coefficients[i])/(x_points[i+1]-x_points[i]);
    d[i] = f_terms[i]*0.5*(x_points[i+1]-x_points[i-1]);

  }
  a[0] = 0; b[0] = 1.0; c[0] = 0.0; d[0] = u_0;
  a[num_grid_points-1] = 0; b[num_grid_points-1] = 1.0; c[num_grid_points-1] = 0.0; d[num_grid_points-1] = u_1;

  // TDMA
  size_t n = num_grid_points-1;
  c[0] = c[0]/b[0];
  d[0] = d[0]/b[0];

  for (size_t i = 1; i < n; i++) {

    c[i] = c[i]/(b[i] - a[i]*c[i-1]);
    d[i] = (d[i] - a[i]*d[i-1]) / (b[i] - a[i]*c[i-1]);

  }

  d[n] = (d[n] - a[n]*d[n-1]) / (b[n] - a[n]*c[n-1]);

  for (int i = n; i-- > 0;) {
    d[i] -= c[i]*d[i+1];
  }

  return d[num_grid_points/2];
}
