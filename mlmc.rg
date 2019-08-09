import 'regent'

-------------------------------------------------------------------------------
-- import libraries and functions, define constants, define terra functions
-------------------------------------------------------------------------------

local C = terralib.includecstring[[
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
]]

local SIM = terralib.includec('diffusion.h')
terralib.linklibrary('libdiffusion.so')

local pow = regentlib.pow(double)
local sqrt = regentlib.sqrt(double)
local ceil = regentlib.ceil(double)

local NUM_LEVELS = 3
local NUM_UNCERTAINTIES = 1
local SEED = 1237
local MAX_SAMPLES_PER_LEVEL = 100
local MAX_ITERS = 10
local TOLERANCE = 0.01

-- Enumeration of states that a sample can be in.
local State = {
  INACTIVE = 0,
  ACTIVE = 1,
  COMPLETED = 2,
}

-- local NUM_U_INPUT = 200000

-- Functions to read in uncertainties from file
--terra read_header(f : &C.FILE)
--  var x: uint64
--  return C.fscanf(f, "%llu\n", &x)
--end
--
--terra read_line(f : &C.FILE, value : &double)
--  return C.fscanf(f, "%lf\n", &value[0]) == 1
--end

-------------------------------------------------------------------------------
-- Tasks
-------------------------------------------------------------------------------

-- Field space: defines the set of "columns" of the samples region; every
-- element of that region will contain a `level` value, a `response` value etc.
local fspace Sample {
  state : int8;
  level : int;
  mesh_size_l : int;
  mesh_size_l_1 : int;
  uncertainties : double[NUM_UNCERTAINTIES];
  response : double;
}

-- This task accepts an arbitrary-size collection of samples and computes the
-- associated simulation response (if it hasn't already been computed).
local task eval_samples(samples : region(ispace(int2d),Sample))
where
  reads(samples.{level, mesh_size_l, mesh_size_l_1, uncertainties}),
  writes(samples.response),
  reads writes(samples.state)
do
  for s in samples do
    if s.state == State.ACTIVE then
      -- Run the simulation once or twice, depending on the sample's level.
      C.printf('lvl %d, uncertainty %e\n', s.level, s.uncertainties[0])
      if s.level > 0 then
        var q_l =
          SIM.diffusion_1d(s.mesh_size_l, NUM_UNCERTAINTIES, s.uncertainties)
        var q_l_1 =
          SIM.diffusion_1d(s.mesh_size_l_1, NUM_UNCERTAINTIES, s.uncertainties)
        s.response = q_l - q_l_1
      else
        s.response =
          SIM.diffusion_1d(s.mesh_size_l, NUM_UNCERTAINTIES, s.uncertainties)
      end
      s.state = State.COMPLETED
    end
  end
end

local task calc_mean(samples : region(ispace(int2d),Sample)) : double
where
  reads(samples.{state, response})
do
  var acc = 0.0
  var count = 0
  for s in samples do
    if s.state == State.COMPLETED then
      acc += s.response
      count += 1
    end
  end
  return acc / count
end

local task calc_var(samples : region(ispace(int2d),Sample),
                    mean : double, num_samples: int) : double
where
  reads(samples.{state, response})
do
  var acc = 0.0
  var count = 0
  for s in samples do
    if s.state == State.COMPLETED then
      acc += pow(s.response, 2)
    end
  end
  return acc/num_samples-pow(mean,2)
end

-------------------------------------------------------------------------------
-- Main
-------------------------------------------------------------------------------

task main()
  __fence(__execution, __block)
  var t_start = regentlib.c.legion_get_current_time_in_micros()
  __fence(__execution, __block)

  -- initialize random seed for uncertain parameter (if not reading inputs from file)
  C.srand48(SEED)

  -- Inputs
  var opt_samples : int[NUM_LEVELS] = array(10,10,10)
  var mesh_sizes : int[NUM_LEVELS] = array(3, 5, 9)
  var q_costs : double[NUM_LEVELS] = array(1.0,2.0,4.0)
  var y_costs : double[NUM_LEVELS] =
    array(q_costs[0],
          q_costs[1],
          q_costs[2])

  -- Algorithm state
  var num_samples : int[NUM_LEVELS] = array(0,0,0)
  var y_mean : double[NUM_LEVELS]
  var y_var : double[NUM_LEVELS]

  -- Define index space, region, and partitions
  var index_space = ispace(int2d,{NUM_LEVELS,MAX_SAMPLES_PER_LEVEL})
  var samples = region(index_space, Sample)
  fill(samples.state, State.INACTIVE)
  var color_space_fine = ispace(int2d,{NUM_LEVELS,MAX_SAMPLES_PER_LEVEL})
  var p_samples_fine = partition(equal, samples, color_space_fine)
  var color_space_by_level = ispace(int2d,{NUM_LEVELS,1})
  var p_samples_by_level = partition(equal, samples, color_space_by_level)

  -- Main loop
  -- Read in uncertaint parameter (if reading uncertainties from file)
  --var f : &C.FILE
  --f = C.fopen("uncertainties_200000.txt", "rb")
  --read_header(f)
  --var uncertainties : double[NUM_U_INPUT]
  --for i=0, NUM_U_INPUT do
  --  var value : double[1]
  --  regentlib.assert(read_line(f, value), "Less data than expected")
  --  uncertainties[i]=value[0]
  --end
  --C.fclose(f)
  
  var iter_print = 0
  for iter = 0, MAX_ITERS do
    -- Run remaining samples for all levels.
    for lvl = 0, NUM_LEVELS do
      for i = num_samples[lvl], opt_samples[lvl] do
        -- Fill in the details for an additional sample.
        samples[{lvl,i}].state = State.ACTIVE
        samples[{lvl,i}].level = lvl
        samples[{lvl,i}].mesh_size_l = mesh_sizes[lvl]
        if lvl > 0 then
          samples[{lvl,i}].mesh_size_l_1 = mesh_sizes[lvl-1]
        end
        for j = 0, NUM_UNCERTAINTIES do
          samples[{lvl,i}].uncertainties[j] = C.drand48()
          --samples[{lvl,i}].uncertainties[j] = uncertainties[lvl*MAX_SAMPLES_PER_LEVEL + i] --not set up for NUM_UNCERTAINTIES > 0 yet
          C.printf('lvl %d, i %d, uncertainty %e\n', lvl, i, samples[{lvl, i}].uncertainties[j]) 
        end
        eval_samples(p_samples_fine[{lvl,i}])
      end
      num_samples[lvl] max= opt_samples[lvl]
    end
    -- Update estimates for central moments.
    for lvl = 0, NUM_LEVELS do
      y_mean[lvl] = calc_mean(p_samples_by_level[{lvl,0}])
      y_var[lvl] = calc_var(p_samples_by_level[{lvl,0}], y_mean[lvl], num_samples[lvl])
      y_var[lvl] max= 0.0 
    end
    -- Update estimates for the optimal number of samples.
    var c = 0.0
    for lvl = 0, NUM_LEVELS do
      c += sqrt(y_costs[lvl] * y_var[lvl])
    end
    c /= pow(TOLERANCE,2)/2.0
    for lvl = 0, NUM_LEVELS do
      opt_samples[lvl] =
        [int](ceil(c * sqrt(y_var[lvl] / y_costs[lvl])))
      regentlib.assert(opt_samples[lvl] < MAX_SAMPLES_PER_LEVEL,
                       'Please increase MAX_SAMPLES_PER_LEVEL')
    end
    -- Print output for this iteration
    --C.printf('Iteration %d:\n', iter)
    --C.printf('  y_costs =')
    --for lvl = 0, NUM_LEVELS do
    --  C.printf(' %e', y_costs[lvl])
    --end
  
    --C.printf('\n')
    --C.printf('  y_mean =')
    --for lvl = 0, NUM_LEVELS do
    --  C.printf(' %e', y_mean[lvl])
    --end
    --C.printf('\n')
    --C.printf('  y_var =')
    --for lvl = 0, NUM_LEVELS do
    --  C.printf(' %e', y_var[lvl])
    --end
    --C.printf('\n')
    --C.printf('  Nl =')
    --for lvl = 0, NUM_LEVELS do
    --  C.printf(' %d', opt_samples[lvl])
    --end
    --C.printf('\n')

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
    iter_print += 1
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

  -- print final results
  __fence(__execution, __block)
  C.printf('iterations = %d \n', iter_print+1)
  C.printf('num_samples =')
  for lvl = 0, NUM_LEVELS do
    C.printf(' %d', num_samples[lvl])
  end
  C.printf('\n')
  C.printf('opt_samples =')
  for lvl = 0, NUM_LEVELS do
    C.printf(' %d', opt_samples[lvl])
  end
  C.printf('\n')
  C.printf('MLMC mean: %e\n', ml_mean)
  C.printf('MLMC std dev: %e\n', sqrt(ml_var))
  --C.printf('MLMC cov: %e\n', sqrt(ml_var)/ml_mean)

  __fence(__execution, __block)
  var t_end = regentlib.c.legion_get_current_time_in_micros()
  C.printf("time (ms): %d\n", (t_end - t_start)/1000)
  __fence(__execution, __block)
end

regentlib.start(main)
