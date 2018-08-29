import 'regent'

-------------------------------------------------------------------------------
-- Imports
-------------------------------------------------------------------------------

local C = terralib.includecstring[[
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
]]

local SIM = terralib.includec('diffusion.h')
terralib.linklibrary('libdiffusion.so')

local pow = regentlib.pow(double)
local sqrt = regentlib.sqrt(double)

-------------------------------------------------------------------------------
-- Constants & inputs
-------------------------------------------------------------------------------

local NUM_LEVELS = 5
local NUM_UNCERTAINTIES = 10
local SEED = 1237
local MAX_SAMPLES_PER_LEVEL = 100
local MAX_ITERATIONS = 5
local TOLERANCE = 0.1

-------------------------------------------------------------------------------
-- Target simulation
-------------------------------------------------------------------------------

task run_sim(mesh_size : int,
             uncertainties : double[NUM_UNCERTAINTIES]) : double
  return SIM.diffusion_1d(mesh_size, NUM_UNCERTAINTIES, uncertainties)
end

-------------------------------------------------------------------------------
-- Main
-------------------------------------------------------------------------------







task main()
  -- Initialize RNG
  C.srand48(SEED)
  -- Inputs
  var num_samples : int[NUM_LEVELS] = array(20,20,20,20,20)
  var mesh_sizes : int[NUM_LEVELS] = array(4,8,16,32,64)
  var q_costs : double[NUM_LEVELS] = array(1.0,8.0,64.0,512.0,4096.0)
  var y_costs : double[NUM_LEVELS] =
    array(q_costs[0],
          q_costs[1] - q_costs[0],
          q_costs[2] - q_costs[1],
          q_costs[3] - q_costs[2],
          q_costs[4] - q_costs[3])
  -- Algorithm state
  var y : (double[MAX_SAMPLES_PER_LEVEL])[NUM_LEVELS]
  var y_mean : double[NUM_LEVELS]
  var y_var : double[NUM_LEVELS]






  -- Run remaining samples for all levels
  for lvl = 0, NUM_LEVELS do
    for i = 0, num_samples[lvl] do
      -- Generate random uncertainties, uniformly distributed in [-1.0,1.0]
      var uncertainties : double[NUM_UNCERTAINTIES]
      for j = 0, NUM_UNCERTAINTIES do
        uncertainties[j] = C.drand48() * 2.0 - 1.0
      end
      -- Run simulation for each choice of uncertainties, on two levels
      if lvl > 0 then
        var q_l = run_sim(mesh_sizes[lvl], uncertainties)
        var q_l_1 = run_sim(mesh_sizes[lvl-1], uncertainties)
        y[lvl][i] = q_l - q_l_1
      else
        y[lvl][i] = run_sim(mesh_sizes[lvl], uncertainties)
      end
    end
  end
  -- Update estimates for central moments
  for lvl = 0, NUM_LEVELS do
    y_mean[lvl] = 0.0
    for i = 0, num_samples[lvl] do
      y_mean[lvl] += y[lvl][i]
    end
    y_mean[lvl] /= num_samples[lvl]
    y_var[lvl] = 0.0
    for i = 0, num_samples[lvl] do
      y_var[lvl] += pow(y[lvl][i] - y_mean[lvl], 2)
    end
    y_var[lvl] /= num_samples[lvl] - 1
  end
  -- Update estimate for optimal number of samples
  var c = 0.0
  for lvl = 0, NUM_LEVELS do
    c += sqrt(y_costs[lvl] * y_var[lvl])
  end
  c /= pow(TOLERANCE,2)/2.0
  for lvl = 0, NUM_LEVELS do
    num_samples[lvl] =
      [int](C.round(c * sqrt(y_var[lvl] / y_costs[lvl])))
    regentlib.assert(num_samples[lvl] < MAX_SAMPLES_PER_LEVEL, '')
  end














  -- Compute MLMC estimator mean & variance
  var ml_mean = 0.0
  for lvl = 0, NUM_LEVELS do
    ml_mean += y_mean[lvl]
  end
  var ml_var = 0.0
  for lvl = 0, NUM_LEVELS do
    ml_var += y_var[lvl] / num_samples[lvl]
  end





  -- TODO:

  -- full algo:
  -- for pilot sample, split values equally
  -- evaluate pilot sample on all levels
  -- accumulate raw moments (Y, Y^2, ...)
  -- estimate central moments
  -- while DNl > 0, iter < maxIter
  -- Nl = ...
  -- DNl = round(Nl^curr - Nl^prev)
  -- do final moment calculation

  -- run with multiple CPUs, profile

  -- will get future-blocked becase of conservative array access conflicts?
  -- should metaprogram to have diff. variables?
  -- should pass sample info as sub-regions

  -- should group more work in tasks

end

regentlib.start(main)
