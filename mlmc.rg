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
#include <unistd.h>
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

local NUM_LEVELS = 2
local NUM_UNCERTAINTIES = 9
local SEED = 1237
local MAX_SAMPLES_PER_LEVEL = 1000
local MAX_ITERS = 10
local TOLERANCE = 0.001

-- Enumeration of states that a sample can be in.
local State = {
  INACTIVE = 0,
  ACTIVE = 1,
  COMPLETED = 2,
}

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

local function mk_eval_samples(level)
  local task eval_samples(samples : region(ispace(int2d),Sample))
  where
    reads(samples.{level, mesh_size_l, mesh_size_l_1, uncertainties}),
    writes(samples.response),
    reads writes(samples.state)
  do
    for s in samples do
      if s.state == State.ACTIVE then
        -- Run the simulation once or twice, depending on the sample's level.
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
        if s.level == 0 then
          C.usleep(1000)
        elseif s.level == 1 then
          C.usleep(5000)
        else regentlib.assert(false, '') end
      end
    end
  end
  local name = 'eval_samples_'..level
  eval_samples:set_name(name)
  eval_samples:get_primary_variant():get_ast().name[1] = name
  return eval_samples
end
local eval_samples = {}
for lvl = 0, NUM_LEVELS-1 do
  eval_samples[lvl] = mk_eval_samples(lvl)
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
                    mean : double) : double
where
  reads(samples.{state, response})
do
  var acc = 0.0
  var count = 0
  for s in samples do
    if s.state == State.COMPLETED then
      acc += pow(s.response - mean, 2)
      count += 1
    end
  end
  return acc / (count - 1)
end

-------------------------------------------------------------------------------
-- Main
-------------------------------------------------------------------------------

task main()
  -- Initialize RNG.
  C.srand48(SEED)
  -- Inputs
  var opt_samples : int[NUM_LEVELS] = array(10,2)
  var mesh_sizes : int[NUM_LEVELS] = array(4,8)
  var q_costs : double[NUM_LEVELS] = array(1.0,8.0)
  var y_costs : double[NUM_LEVELS] =
    array(q_costs[0],
          q_costs[1] + q_costs[0])
  -- Algorithm state
  var num_samples : int[NUM_LEVELS] = array(0,0)
  var y_mean : double[NUM_LEVELS]
  var y_var : double[NUM_LEVELS]
  -- Region of samples
  -- Index space: controls the number of "rows"; in this case each row is
  -- indexed by a pair of integers, the first defining the level and the
  -- second defining the sample index within that level, for a total of
  -- NUM_LEVELS * MAX_SAMPLES_PER_LEVEL rows.
  var index_space = ispace(int2d,{NUM_LEVELS,MAX_SAMPLES_PER_LEVEL})
  var samples = region(index_space, Sample)
  fill(samples.state, State.INACTIVE)
  -- Split the region of samples into a disjoint set of sub-regions, such that
  -- each of those sub-regions can be operated on independently from the rest.
  -- The first partition splits the region into as many parts as it has
  -- elements, therefore each sub-region will contain a single sample; this is
  -- usually not ideal, because it means we will end up launching many tasks,
  -- each doing very little work.
  var color_space_fine = ispace(int2d,{NUM_LEVELS,MAX_SAMPLES_PER_LEVEL})
  var p_samples_fine = partition(equal, samples, color_space_fine)
  -- The same region can be partition in multiple ways; when switching from
  -- using one partition to the other, the runtime will automatically check if
  -- the two uses conflict with each other.
  -- Our second partition splits the region by level; it splits its first
  -- dimension (the level index) into NUM_LEVELS parts, and doesn't split its
  -- second dimension. Therefore, there will be a total of NUM_LEVELS pieces:
  -- {0,0} {1,0} ... {NUM_LEVELS-1,0}
  var color_space_by_level = ispace(int2d,{NUM_LEVELS,1})
  var p_samples_by_level = partition(equal, samples, color_space_by_level)
  -- Main loop
  var iter = 0
  for iter = 0, MAX_ITERS do
    -- Create remaining samples for all levels.
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
          samples[{lvl,i}].uncertainties[j] = C.drand48() * 2.0 - 1.0
        end
      end
    end
    -- Run remaining samples for all levels.
    for lvl = NUM_LEVELS-1, -1, -1 do
      for i = num_samples[lvl], opt_samples[lvl] do
        -- Invoke `eval_samples` for a set of samples containing just the
        -- newly created sample. This task will be launched asynchronously; the
        -- main task will continue running and launching new tasks. The runtime
        -- will analyze the dependencies between the newly launched task and
        -- all tasks launched before it. It will conclude that the new task is
        -- independent from all the others, and thus can be queued to run
        -- immediately.
        if lvl == 0 then
          [eval_samples[0]](p_samples_fine[{lvl,i}])
        elseif lvl == 1 then
          [eval_samples[1]](p_samples_fine[{lvl,i}])
        else
          regentlib.assert(false, '')
        end
      end
      num_samples[lvl] max= opt_samples[lvl]
    end
    -- Update estimates for central moments.
    for lvl = 0, NUM_LEVELS do
      -- At this point we switch to using the partition by level; the runtime
      -- will analyze the dependencies and conclude that the call to
      -- `calc_mean` has a read-after-write dependency with all the preceding
      -- calls to `eval_samples` for samples on the same level. Therefore, the
      -- execution of `calc_mean` will have to wait until those tasks have
      -- completed (the main task is free to continue emitting tasks, however).
      y_mean[lvl] = calc_mean(p_samples_by_level[{lvl,0}])
      y_var[lvl] = calc_var(p_samples_by_level[{lvl,0}], y_mean[lvl])
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
