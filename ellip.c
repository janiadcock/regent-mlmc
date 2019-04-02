// Code by Lluis Jofre Cruanyes

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
// Here we set f = -1 and k is a random diffusivity


double ellip_1d(size_t num_grid_points,size_t num_uncertainties,double *xi_uncertainties) {

  // Problem parameters
  double domain_length = 1.0;		// Length of the domain (starts at 0)
  double u_0 = 0.0;			// Set left boundary value for unknowns
  double u_1 = 0.0;			// Set right boundary value for unknowns
  double f = -1.0;			// Set forcing term value
  double sigma = 1.0;			// Set variability of diffusivity
  double grid_spacing;

  // Generate grid array
  double x_points[num_grid_points];	// Includes boundary points
  for(size_t i = 0; i < num_grid_points; i++) {

    // Grid spacing
    grid_spacing = domain_length/(num_grid_points-1);

    // Set points
    x_points[i] = i*grid_spacing;
    printf("i: %zu, x_points[i]: %f \n", i, x_points[i]);

  }


  // Generate & set forcing term array
  double f_terms[num_grid_points];	// Includes boundary points
  for(size_t i = 0; i < num_grid_points; i++) {

    // Initialize
    f_terms[i] = f;

  }


  // Generate & compute stochastic coefficient array
  double k_coefficients[num_grid_points];	// Includes boundary points
  for(size_t i = 0; i < num_grid_points; i++) {

    // Initialize
    k_coefficients[i] = 1.0;
    for(size_t k = 0; k < num_uncertainties; k++) {

      k_coefficients[i] += sigma*((1.0/((k+1.0)*(k+1.0)*PI*PI))*cos(2.0*PI*(k+1.0)*x_points[i])*xi_uncertainties[k]);
      printf("i: %zu, k_coefficients[i]: %f \n", i, k_coefficients[i]);
    }

  }


  // Solve linear system of equations using TDMA (boundary points solved using identity)
  double a[num_grid_points];	// Includes boundary points
  double b[num_grid_points];	// Includes boundary points
  double c[num_grid_points];	// Includes boundary points
  double d[num_grid_points];	// Includes boundary points
  for(size_t i = 1; i < (num_grid_points-1); i++) {

    a[i] = ( 0.5*(k_coefficients[i]+k_coefficients[i-1])/(x_points[i]-x_points[i-1]) )/( 0.5*(x_points[i+1]-x_points[i-1]) );
    b[i] = ( (-1.0)*0.5*(k_coefficients[i+1]+k_coefficients[i])/(x_points[i+1]-x_points[i])+(-1.0)*0.5*(k_coefficients[i]+k_coefficients[i-1])/(x_points[i]-x_points[i-1]) )/( 0.5*(x_points[i+1]-x_points[i-1]) );
    c[i] = ( 0.5*(k_coefficients[i+1]+k_coefficients[i])/(x_points[i+1]-x_points[i]) )/( 0.5*(x_points[i+1]-x_points[i-1]) );
    d[i] = f_terms[i];
    //printf("i: %zu, d[i]: %f \n", i, d[i]);
  }
  a[0] = 0; b[0] = 1.0; c[0] = 0.0; d[0] = u_0;
  a[num_grid_points-1] = 0; b[num_grid_points-1] = 1.0; c[num_grid_points-1] = 0.0; d[num_grid_points-1] = u_1;

  // TDMA
  double c_star[num_grid_points];	// Includes boundary points
  double d_star[num_grid_points];	// Includes boundary points

  c_star[0] = c[0]/b[0];
  d_star[0] = d[0]/b[0];

  double m = 0.0;
  for(size_t i = 1; i < (num_grid_points-1); i++) {

    m = 1.0/(b[i]-a[i]*c_star[i-1]);
    c_star[i] = c[i]*m;
    d_star[i] = (d[i]-a[i]*d_star[i-1])*m;
    //printf("i: %zu, d_star[i]: %f \n", i, d_star[i]);
  }
  m = 1.0/(b[num_grid_points-1]-a[num_grid_points-1]*c_star[num_grid_points-2]);
  d_star[num_grid_points-1] = (d[num_grid_points-1]-a[num_grid_points-1]*d_star[num_grid_points-2])*m;

  double u_unknowns[num_grid_points];	// Includes boundary points
  u_unknowns[num_grid_points-1] = d_star[num_grid_points-1];
  for(int i = (num_grid_points-2); i > -1; i--) {

    u_unknowns[i] = d_star[i]-c_star[i]*u_unknowns[i+1];
    printf("i: %d, u_unknowns[i]: %f \n", i, u_unknowns[i]);
  }
  printf("i: %zu, u_unknowns[%zu]: %f \n", num_grid_points-1, num_grid_points-1, u_unknowns[num_grid_points-1]);
  printf("num_grid_point: %zu \n", num_grid_points);
  printf("grid_spacing: %f \n", grid_spacing);
  if (num_grid_points == 3) {printf("uncertainty: %f", xi_uncertainties[0]);}
  double sum = 0; 
  for (size_t i=0; i<num_grid_points; i++) {sum += u_unknowns[i]*grid_spacing;}
  return sum;
  //return u_unknowns[num_grid_points/2];

}
