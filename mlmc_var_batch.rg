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
local ceil = regentlib.ceil(double)
local floor = regentlib.floor(double)
local log = regentlib.log(double)
-------------------------------------------------------------------------------
-- Constants & inputs
-------------------------------------------------------------------------------

--local NUM_LEVELS = 3
local NUM_UNCERTAINTIES = 1
local SEED = 1237
local MAX_SAMPLES_PER_LEVEL = 1000
local MAX_ITERS = 10
local TOLERANCE = 0.000005

local NUM_U_INPUT = 100000

local NUM_REPLICATES = 10
-- Enumeration of states that a sample can be in.
local State = {
  INACTIVE = 0,
  ACTIVE = 1,
  COMPLETED = 2,
}

-------------------------------------------------------------------------------
-- Tools to read in uncertainties from file

terra read_header(f : &C.FILE)
  var x: uint64
  return C.fscanf(f, "%llu\n", &x)
end

terra read_line(f : &C.FILE, value : &double)
  return C.fscanf(f, "%lf\n", &value[0]) == 1
end
-------------------------------------------------------------------------------

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
  response_l : double;
  response_l_1 : double;
  response : double;
  response_MC : double; 
}

-- This task accepts an arbitrary-size collection of samples and computes the
-- associated simulation response (if it hasn't already been computed).
local task eval_samples(samples : region(ispace(int2d),Sample))
where
  reads(samples.{level, mesh_size_l, mesh_size_l_1, uncertainties}),
  writes(samples.response_l, samples.response_l_1),
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
        s.response_l = q_l
        s.response_l_1 = q_l_1
      else
        s.response_l =
          SIM.diffusion_1d(s.mesh_size_l, NUM_UNCERTAINTIES, s.uncertainties)
      end
      s.state = State.COMPLETED
    end
  end
end

local task calc_response(samples : region(ispace(int2d), Sample), q_l_mean : double, q_l_1_mean : double)
where
  reads(samples.{state, level, response_l, response_l_1}),
  writes(samples.response)
do
  for s in samples do
    if s.state == State.COMPLETED then
      if s.level > 0 then
        s.response = pow(s.response_l - q_l_mean, 2) - pow(s.response_l_1 - q_l_1_mean, 2)
      else
        s.response = pow(s.response_l - q_l_mean, 2)
      end
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

local task calc_mean_l(samples : region(ispace(int2d),Sample)) : double
where
  reads(samples.{state, response_l})
do
  var acc = 0.0
  var count = 0
  for s in samples do
    if s.state == State.COMPLETED then
      acc += s.response_l
      count += 1
    end
  end
  return acc / count
end

local task calc_mean_l_sq(samples: region(ispace(int2d), Sample)): double
where 
  reads(samples.{state, response_l})
do
  var acc = 0.0
  var count = 0
  for s in samples do
    if s.state == State.COMPLETED then
      acc += pow(s.response_l, 2)
      count += 1
    end
  end
  return acc / count
end


local task calc_mean_l_1(samples : region(ispace(int2d),Sample)) : double
where
  reads(samples.{state, response_l_1})
do
  var acc = 0.0
  var count = 0
  for s in samples do
    if s.state == State.COMPLETED then
      acc += s.response_l_1
      count += 1
    end
  end
  return acc / count
end


local task calc_var(samples : region(ispace(int2d),Sample), mean : double) : double
where
  reads(samples.{state, response})
do
  var acc = 0.0
  var count = 0
  for s in samples do
    if s.state == State.COMPLETED then
      acc += pow(s.response, 2)
      count += 1
    end
  end
  return acc/count-pow(mean,2)
end

local task calc_response_MC(samples : region(ispace(int2d), Sample), q_l_mean : double)
where
  reads(samples.{state, level, response_l}),
  writes(samples.response_MC)
do
  for s in samples do
    if s.state == State.COMPLETED then
      s.response_MC = pow(s.response_l - q_l_mean, 2)
    end
  end
end

local task calc_mean_MC(samples : region(ispace(int2d),Sample)) : double
where
  reads(samples.{state, response_MC})
do
  var acc = 0.0
  var count = 0
  for s in samples do
    if s.state == State.COMPLETED then
      acc += s.response_MC
      count += 1
    end
  end
  return acc / count
end

local task calc_var_MC(samples : region(ispace(int2d),Sample), mean_MC : double): double
where
  reads(samples.{state, response_MC})
do
  var acc = 0.0
  --var acc_2 = 0.0
  var count = 0
  for s in samples do
    if s.state == State.COMPLETED then
      acc += pow(s.response_MC, 2)
      --acc_2 += pow(s.response_MC - mean_MC, 2)
      count += 1
    end
  end
  return acc/count - pow(mean_MC, 2)
end

-- Main
-------------------------------------------------------------------------------

task main()
  -----
  --Calculate alpha, beta, gamma needed for dynamic levels
  C.srand48(SEED)

  --If reading in uncertainties from file
  var f : &C.FILE
  f = C.fopen("uncertainties_large.txt", "rb")
  read_header(f)
  --regentlib.assert(read_header(f) == NUM_U_INPUT, "Input file wrong number of uncertainties.")
  var uncertainties : double[NUM_U_INPUT]
  for i=0, NUM_U_INPUT do
    var value : double[1]
    regentlib.assert(read_line(f, value), "Less data than expected")
    uncertainties[i]=value[0]
  end
  C.fclose(f)
  
  --for i = 0, NUM_U_INPUT do
  --  C.printf("%f\n", uncertainties[i])
  --end

  var alpha = 2.00974
  var beta = 3.998377
  C.printf('alpha: %f \n', alpha)
  C.printf('beta: %f \n', beta)
  C.printf('\n')

  -----
  -- Run MLMC
  -- Initialize RNG.
  C.srand48(SEED)
  -- Inputs
  var M = 2 --refinement cost factor
  var gamma = log(M)/log(2) --using expected gamma not approximated one
  C.printf('expected gamma: %f \n', gamma)
  var NUM_LEVELS = 3
  var MAX_NUM_LEVELS = 10
  var opt_samples : int[MAX_NUM_LEVELS] = array(182,55,10,0,0,0,0,0,0,0)
  var mesh_sizes : int[MAX_NUM_LEVELS] = array(3,5,9,17,33,65,129,257,513,1025)
  var y_costs : double[MAX_NUM_LEVELS] = array(1.0,2.0,4.0,8.0,16.0,32.0,64.0,128.0,256.0,512.0,1024.0)



  -- Algorithm state
  var num_samples : int[MAX_NUM_LEVELS] = array(0,0,0,0,0,0,0,0,0,0)
  var y_mean : double[MAX_NUM_LEVELS] = array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  var y_var : double[MAX_NUM_LEVELS] = array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  -- Region of samples
  -- Index space: controls the number of "rows"; in this case each row is
  -- indexed by a pair of integers, the first defining the level and the
  -- second defining the sample index within that level, for a total of
  -- NUM_LEVELS * MAX_SAMPLES_PER_LEVEL rows.
  var index_space = ispace(int2d,{MAX_NUM_LEVELS,MAX_SAMPLES_PER_LEVEL})
  var samples = region(index_space, Sample)
  fill(samples.state, State.INACTIVE)
  -- Split the region of samples into a disjoint set of sub-regions, such that
  -- each of those sub-regions can be operated on independently from the rest.
  -- The first partition splits the region into as many parts as it has
  -- elements, therefore each sub-region will contain a single sample; this is
  -- usually not ideal, because it means we will end up launching many tasks,
  -- each doing very little work.
  var color_space_fine = ispace(int2d,{MAX_NUM_LEVELS,MAX_SAMPLES_PER_LEVEL})
  var p_samples_fine = partition(equal, samples, color_space_fine)
  -- The same region can be partition in multiple ways; when switching from
  -- using one partition to the other, the runtime will automatically check if
  -- the two uses conflict with each other.
  -- Our second partition splits the region by level; it splits its first
  -- dimension (the level index) into NUM_LEVELS parts, and doesn't split its
  -- second dimension. Therefore, there will be a total of NUM_LEVELS pieces:
  -- {0,0} {1,0} ... {NUM_LEVELS-1,0}
  var color_space_by_level = ispace(int2d,{MAX_NUM_LEVELS,1})
  var p_samples_by_level = partition(equal, samples, color_space_by_level)


  --Replicates loop
  var var_MLMC_replicates : double[NUM_REPLICATES]
  var var_MC_replicates : double[NUM_REPLICATES]
  var uncertainty_i = 0
  for k = 0, NUM_REPLICATES do
    C.printf('replicate: %d \n', k)
    --C.printf('num uncertanties used: %d \n', uncertainty_i)
    --reset samples
    opt_samples[0] = 182
    opt_samples[1] = 55
    opt_samples[2] = 10
    for i = 3, MAX_NUM_LEVELS do
      opt_samples[i] = 0   
    end
    for i = 0, MAX_NUM_LEVELS do
      num_samples[i] = 0
      y_mean[i] = 0.0
      y_var[i] = 0.0
    end
    fill(samples.state, State.INACTIVE)
    NUM_LEVELS = 3

    -- Main loop
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
          --samples[{lvl,i}].uncertainties[j] = .5
          --samples[{lvl,i}].uncertainties[j] = C.drand48()
          samples[{lvl,i}].uncertainties[j] = uncertainties[uncertainty_i] --not set up for NUM_UNCERTAINTIES > 0 yet
          uncertainty_i += 1
        end
        -- Invoke `eval_samples` for a set of samples containing just the
        -- newly created sample. This task will be launched asynchronously; the
        -- main task will continue running and launching new tasks. The runtime
        -- will analyze the dependencies between the newly launched task and
        -- all tasks launched before it. It will conclude that the new task is
        -- independent from all the others, and thus can be queued to run
        -- immediately.
        eval_samples(p_samples_fine[{lvl,i}])
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
      var q_l_mean = C.fabs(calc_mean_l(p_samples_by_level[{lvl,0}]))
      var q_l_1_mean = 0.0
      if lvl > 0 then
        q_l_1_mean = C.fabs(calc_mean_l_1(p_samples_by_level[{lvl,0}]))
      end
      calc_response(p_samples_by_level[{lvl,0}], q_l_mean, q_l_1_mean)
      
      y_mean[lvl] = C.fabs(calc_mean(p_samples_by_level[{lvl,0}]))
      y_var[lvl] = calc_var(p_samples_by_level[{lvl,0}], y_mean[lvl])
      y_var[lvl] max= 0.0 
    end
  
    --cope with possible zero values for ml and Vl
    --(can happen when there are few samples, e.g. on the final level added)
    for lvl = 2, NUM_LEVELS do
      y_mean[lvl] max= 0.5*y_mean[lvl-1]/pow(2, alpha)
      y_var[lvl] max= 0.5*y_var[lvl-1]/pow(2, beta)
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
    --C.printf('MLMC cov: %e\n', sqrt(ml_var)/ml_mean)
  
    -- Comparison to MC on finest level w/ same computational cost
    var total_C = 0.0
    for lvl = 0, NUM_LEVELS do
      total_C += num_samples[lvl]* y_costs[lvl]
    end
    
    var lvl = NUM_LEVELS-1
    --var N_L = floor(total_C/y_costs[lvl])
    var N_L = (int) (total_C/y_costs[lvl])
    C.printf('MLMC total cost %e \n', total_C)
    C.printf('MC number of samples %d \n', N_L)
  
    if N_L > MAX_SAMPLES_PER_LEVEL then
      C.printf('number of samples for MC exceeds MAX_SAMPLES_PER_LEVEL, results shown for MC with MAX_SAMPLES_PER_LEVEL\n')
      N_L = MAX_SAMPLES_PER_LEVEL
    end
  
    for i = num_samples[NUM_LEVELS-1], N_L do
      --Fill in details for additional samples needed for MC
      samples[{lvl,i}].state = State.ACTIVE
      samples[{lvl,i}].level = lvl
      samples[{lvl,i}].mesh_size_l = mesh_sizes[lvl]
      samples[{lvl,i}].mesh_size_l_1 = mesh_sizes[lvl-1]
      for j=0,NUM_UNCERTAINTIES do
        samples[{lvl,i}].uncertainties[j] = uncertainties[uncertainty_i] --not set up for NUM_UNCERTAINTIES > 0 yet
        uncertainty_i += 1
      end
      eval_samples(p_samples_fine[{lvl,i}])
    end
    var q_l_mean = C.fabs(calc_mean_l(p_samples_by_level[{lvl,0}]))
    calc_response_MC(p_samples_by_level[{lvl,0}], q_l_mean)
    var mean_MC_L = C.fabs(calc_mean_MC(p_samples_by_level[{lvl,0}]))
    var var_MC_L = calc_var_MC(p_samples_by_level[{lvl, 0}], mean_MC_L)
    C.printf('MC mean: %e\n', mean_MC_L)
    C.printf('MC stddev: %e\n', sqrt(var_MC_L))
    var_MLMC_replicates[k] = ml_mean
    var_MC_replicates[k] = mean_MC_L


    C.printf('opt_samples =')
    for lvl = 0, NUM_LEVELS do
      C.printf(' %d', opt_samples[lvl])
    end
    C.printf('\n')


    C.printf('num_samples =')
    for lvl = 0, NUM_LEVELS do
      C.printf(' %d', num_samples[lvl])
    end
    C.printf('\n')

    C.printf('\n')
  end

  var acc_MLMC = 0.0
  var acc_MC = 0.0
  var acc_MLMC_sq = 0.0
  var acc_MC_sq = 0.0
  var count = 0
  for i = 0, NUM_REPLICATES do
    acc_MLMC += var_MLMC_replicates[i]
    acc_MC += var_MC_replicates[i]
    acc_MLMC_sq += pow(var_MLMC_replicates[i], 2)
    acc_MC_sq += pow(var_MC_replicates[i], 2)
    count += 1
  end
  var MLMC_replicate_mean = acc_MLMC/count
  var MC_replicate_mean = acc_MC/count
  var MLMC_replicate_var = acc_MLMC_sq/count - pow(MLMC_replicate_mean, 2)
  var MC_replicate_var = acc_MC_sq/count - pow(MC_replicate_mean, 2)
  
  C.printf('MLMC sample replicate mean: %e\n', MLMC_replicate_mean)
  C.printf('MC sample replicate mean: %e\n', MC_replicate_mean)
  C.printf('MLMC sample replicate variance: %e\n', MLMC_replicate_var)
  C.printf('MC sample replicate variance: %e\n', MC_replicate_var)

  C.printf('MLMC sample replicates: \n')
  for i = 0, NUM_REPLICATES do
    C.printf('%.1e ', var_MLMC_replicates[i])
  end
  C.printf('\n')


  C.printf('MC sample replicates: \n')
  for i = 0, NUM_REPLICATES do
    C.printf('%.1e ', var_MC_replicates[i])
  end
  C.printf('\n')

end
regentlib.start(main)
