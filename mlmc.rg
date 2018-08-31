import 'regent'

-------------------------------------------------------------------------------
-- Imports
-------------------------------------------------------------------------------

-- These headers are parsed, and the symbols they define become available to
-- the code below. When the code is executed, these symbols will be looked up
-- in the set of linked files.
local C = terralib.includecstring[[
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
]]

-- Similar to above, we parse our simulation's header file to import the name
-- of our simulation kernel.
local SIM = terralib.includec('diffusion.h')
-- We also dynamically link our simulation's code into the current process,
-- because by default the Regent compiler will compile our tasks and load them
-- into the current process to be executed.
terralib.linklibrary('libdiffusion.so')

-- Load some built-in math functions.
local pow = regentlib.pow(double)
local sqrt = regentlib.sqrt(double)

-------------------------------------------------------------------------------
-- Constants & inputs
-------------------------------------------------------------------------------

local NUM_LEVELS = 5
local NUM_UNCERTAINTIES = 10
local SEED = 1237
local MAX_SAMPLES_PER_LEVEL = 1000
local MAX_ITERS = 10
local TOLERANCE = 0.01

-------------------------------------------------------------------------------
-- Target simulation
-------------------------------------------------------------------------------

task run_sim(mesh_size : int,
             uncertainties : double[NUM_UNCERTAINTIES]) : double
  return SIM.diffusion_1d(mesh_size, NUM_UNCERTAINTIES, uncertainties)
end

task run_lvl_0(y : region(ispace(int2d), double),
               mesh_size : int,
               uncertainties : double[NUM_UNCERTAINTIES])
where writes(y) do
  for i in y do
    y[i] = run_sim(mesh_size, uncertainties)
  end
end

task run_lvl_1_plus(y : region(ispace(int2d), double),
                    mesh_size_l : int,
                    mesh_size_l_1 : int,
                    uncertainties : double[NUM_UNCERTAINTIES])
where writes(y) do
  for i in y do
    var q_l = run_sim(mesh_size_l, uncertainties)
    var q_l_1 = run_sim(mesh_size_l_1, uncertainties)
    y[i] = q_l - q_l_1
  end
end

-------------------------------------------------------------------------------
-- Main
-------------------------------------------------------------------------------

task main()
  -- Initialize RNG.
  C.srand48(SEED)
  -- Inputs
  var opt_samples : int[NUM_LEVELS] = array(20,20,20,20,20)
  var mesh_sizes : int[NUM_LEVELS] = array(4,8,16,32,64)
  var q_costs : double[NUM_LEVELS] = array(1.0,8.0,64.0,512.0,4096.0)
  var y_costs : double[NUM_LEVELS] =
    array(q_costs[0],
          q_costs[1] + q_costs[0],
          q_costs[2] + q_costs[1],
          q_costs[3] + q_costs[2],
          q_costs[4] + q_costs[3])
  -- Algorithm state
  var num_samples : int[NUM_LEVELS] = array(0,0,0,0,0)
  var y_mean : double[NUM_LEVELS]
  var y_var : double[NUM_LEVELS]
  var is = ispace(int2d,{NUM_LEVELS,MAX_SAMPLES_PER_LEVEL})
  var y = region(is, double)
  var p_y = partition(equal, y, is)
  -- Main loop
  var iter = 0
  for iter = 0, MAX_ITERS do
    -- Run remaining samples for all levels.
    for lvl = 0, NUM_LEVELS do
      for i = num_samples[lvl], opt_samples[lvl] do
        -- Generate random uncertainties, uniformly distributed in [-1.0,1.0]
        var uncertainties : double[NUM_UNCERTAINTIES]
        for j = 0, NUM_UNCERTAINTIES do
          uncertainties[j] = C.drand48() * 2.0 - 1.0
        end
        -- Run simulation for each choice of uncertainties, on two levels
        if lvl > 0 then
          run_lvl_1_plus(p_y[{lvl,i}],
                         mesh_sizes[lvl],
                         mesh_sizes[lvl-1],
                         uncertainties)
        else
          run_lvl_0(p_y[{lvl,i}],
                    mesh_sizes[lvl],
                    uncertainties)
        end
      end
      num_samples[lvl] max= opt_samples[lvl]
    end
    -- Update estimates for central moments.
    for lvl = 0, NUM_LEVELS do
      y_mean[lvl] = 0.0
      for i = 0, num_samples[lvl] do
        y_mean[lvl] += y[{lvl,i}]
      end
      y_mean[lvl] /= num_samples[lvl]
      y_var[lvl] = 0.0
      for i = 0, num_samples[lvl] do
        y_var[lvl] += pow(y[{lvl,i}] - y_mean[lvl], 2)
      end
      y_var[lvl] /= num_samples[lvl] - 1
    end
    -- Update estimates for the optimal number of samples.
    var c = 0.0
    for lvl = 0, NUM_LEVELS do
      c += sqrt(y_costs[lvl] * y_var[lvl])
    end
    c /= pow(TOLERANCE,2)/2.0
    for lvl = 0, NUM_LEVELS do
      opt_samples[lvl] =
        [int](C.round(c * sqrt(y_var[lvl] / y_costs[lvl])))
      regentlib.assert(opt_samples[lvl] < MAX_SAMPLES_PER_LEVEL,
                       'Please increase MAX_SAMPLES_PER_LEVEL')
    end
    -- Print output.
    C.printf('Iteration %d:\n', iter)
    C.printf('  y_mean =')
    for lvl = 0, NUM_LEVELS do
      C.printf(' %e', y_mean[lvl])
    end
    C.printf('\n')
    C.printf('  y_var =')
    for lvl = 0, NUM_LEVELS do
      C.printf(' %e', y_var[lvl])
    end
    C.printf('\n')
    C.printf('  Nl =')
    for lvl = 0, NUM_LEVELS do
      C.printf(' %d', opt_samples[lvl])
    end
    C.printf('\n')
    -- Decide if we have converged.
    var opt_samples_ran = true
    for lvl = 0, NUM_LEVELS do
      if opt_samples[lvl] > num_samples[lvl] then
        opt_samples_ran = false
        break
      end
    end
    if opt_samples_ran then
      break
    end
  end
  -- Compute MLMC estimator mean & variance.
  var ml_mean = 0.0
  for lvl = 0, NUM_LEVELS do
    ml_mean += y_mean[lvl]
  end
  var ml_var = 0.0
  for lvl = 0, NUM_LEVELS do
    ml_var += y_var[lvl] / num_samples[lvl]
  end
  C.printf('MLMC mean: %e\n', ml_mean)
  C.printf('MLMC stddev: %e\n', sqrt(ml_var))
  C.printf('MLMC cov: %e\n', sqrt(ml_var)/ml_mean)
end

regentlib.start(main)
