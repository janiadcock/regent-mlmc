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
local TOLERANCE = 0.001

local NUM_SAMPLES_CONV = 1000 
local NUM_LEVELS_CONV = 9

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

-- Main
-------------------------------------------------------------------------------

task main()
  -----
  --Calculate alpha, beta, gamma needed for dynamic levels
  C.srand48(SEED)
  var mesh_sizes_conv : int[NUM_LEVELS_CONV] = array(3, 5, 9, 17, 33, 65, 129, 257, 513)
  var cost_conv : double[NUM_LEVELS_CONV]
  var y_mean_conv : double[NUM_LEVELS_CONV]
  var y_var_conv : double[NUM_LEVELS_CONV] 

  var index_space_conv = ispace(int2d,{NUM_LEVELS_CONV,NUM_SAMPLES_CONV})
  var samples_conv = region(index_space_conv, Sample)
  fill(samples_conv.state, State.INACTIVE)
  var color_space_fine_conv = ispace(int2d,{NUM_LEVELS_CONV, NUM_SAMPLES_CONV}) 
  var p_samples_fine_conv = partition(equal, samples_conv, color_space_fine_conv)
  var color_space_by_level_conv = ispace(int2d,{NUM_LEVELS_CONV,1})
  var p_samples_by_level_conv = partition(equal, samples_conv, color_space_by_level_conv)

  for lvl = 0, NUM_LEVELS_CONV do
    C.printf('level: %d \n', lvl)
    __fence(__execution, __block)
    var t_start = regentlib.c.legion_get_current_time_in_micros()
    __fence(__execution, __block)
    for i = 0, NUM_SAMPLES_CONV do
      samples_conv[{lvl,i}].state = State.ACTIVE
      samples_conv[{lvl,i}].level = lvl
      samples_conv[{lvl,i}].mesh_size_l = mesh_sizes_conv[lvl]
      if lvl > 0 then
        samples_conv[{lvl,i}].mesh_size_l_1 = mesh_sizes_conv[lvl-1]
      end
      for j = 0, NUM_UNCERTAINTIES do
        --samples_conv[{lvl,i}].uncertainties[j] = .5
        samples_conv[{lvl,i}].uncertainties[j] = C.drand48()
      end
      eval_samples(p_samples_fine_conv[{lvl,i}])
    end
    __fence(__execution, __block)
    cost_conv[lvl] = (regentlib.c.legion_get_current_time_in_micros() - t_start)*pow(10, -6)
    __fence(__execution, __block)
  end
  for lvl = 0, NUM_LEVELS_CONV do
    y_mean_conv[lvl] = C.fabs(calc_mean(p_samples_by_level_conv[{lvl,0}]))
    --var N_samples_conv = NUM_SAMPLES_CONV 
    y_var_conv[lvl] = calc_var(p_samples_by_level_conv[{lvl,0}], y_mean_conv[lvl])
  end

  --should change into C function
  var min_L = 2
  min_L max= floor(0.4*NUM_LEVELS_CONV)
  var m_x = 0.0
  var alpha_m_y = 0.0
  var beta_m_y = 0.0
  var SS_xx = 0.0
  var alpha_SS_xy = 0.0
  var beta_SS_xy = 0.0
  var count = 0
  for lvl = min_L-1, NUM_LEVELS_CONV do
    var x_i = lvl+1
    var alpha_y_i = log(C.fabs(y_mean_conv[lvl]))/log(2)
    var beta_y_i = log(C.fabs(y_var_conv[lvl]))/log(2)
    m_x += x_i
    alpha_m_y += alpha_y_i
    beta_m_y += beta_y_i
    SS_xx += x_i*x_i
    alpha_SS_xy += x_i*alpha_y_i
    beta_SS_xy += x_i * beta_y_i
    count += 1
  end
  m_x = m_x/count
  alpha_m_y = alpha_m_y/count
  beta_m_y = beta_m_y/count
  SS_xx = SS_xx - count*m_x*m_x
  alpha_SS_xy = alpha_SS_xy - count*m_x*alpha_m_y
  beta_SS_xy = beta_SS_xy - count*m_x*beta_m_y
  var alpha = -alpha_SS_xy/SS_xx
  var beta = -beta_SS_xy/SS_xx
  var gamma = log(cost_conv[NUM_LEVELS_CONV-1]/cost_conv[NUM_LEVELS_CONV-2])/log(2)
  C.printf('alpha: %f \n', alpha)
  C.printf('beta: %f \n', beta)
  C.printf('gamma: %f \n', gamma)
  C.printf('\n')

  -----
  -- Run MLMC
  -- Initialize RNG.
  C.srand48(SEED)
  -- Inputs
  var NUM_LEVELS = 3
  var opt_samples : int[NUM_LEVELS] = array(1000,1000,1000)
  var mesh_sizes : int[NUM_LEVELS] = array(3, 5, 9)
  var q_costs : double[NUM_LEVELS] = array(1.0,2.0,4.0)
  var y_costs : double[NUM_LEVELS] =
    array(q_costs[0],
          q_costs[1],
          q_costs[2])
  -- Algorithm state
  var num_samples : int[NUM_LEVELS] = array(0,0,0)
  var y_mean : double[NUM_LEVELS] = array(0.0, 0.0, 0.0)
  var y_var : double[NUM_LEVELS] = array(0.0, 0.0, 0.0)
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
          samples[{lvl,i}].uncertainties[j] = C.drand48()
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
      C.printf('level: %d \n', lvl)
      C.printf('pre num_samples[lvl]: %d \n', num_samples[lvl])
      num_samples[lvl] max= opt_samples[lvl]
      --C.printf('max: %f', max)
      C.printf('post num_samples[lvl]: %d \n', num_samples[lvl])
      C.printf('opt_samples[lvl]: %d \n', opt_samples[lvl])
    end
    -- Update estimates for central moments.
    for lvl = 0, NUM_LEVELS do
      -- At this point we switch to using the partition by level; the runtime
      -- will analyze the dependencies and conclude that the call to
      -- `calc_mean` has a read-after-write dependency with all the preceding
      -- calls to `eval_samples` for samples on the same level. Therefore, the
      -- execution of `calc_mean` will have to wait until those tasks have
      -- completed (the main task is free to continue emitting tasks, however).
      y_mean[lvl] = C.fabs(calc_mean(p_samples_by_level[{lvl,0}]))
      C.printf('num_samples[lvl] %d \n', num_samples[lvl])
      y_var[lvl] = calc_var(p_samples_by_level[{lvl,0}], y_mean[lvl])
      y_var[lvl] max= 0.0 
    end
    -- Update estimates for the optimal number of samples.
    var c = 0.0
    for lvl = 0, NUM_LEVELS do
      c += sqrt(y_costs[lvl] * y_var[lvl])
    end
    c /= pow(TOLERANCE,2)/2.0
 
    var almost_conv = true
    for lvl = 0, NUM_LEVELS do
      opt_samples[lvl] =
        [int](ceil(c * sqrt(y_var[lvl] / y_costs[lvl])))
      regentlib.assert(opt_samples[lvl] < MAX_SAMPLES_PER_LEVEL,
                       'Please increase MAX_SAMPLES_PER_LEVEL')
      if (opt_samples[lvl] - num_samples[lvl]) > 0.01*num_samples[lvl] then
        almost_conv = false
      end
    end
    -- dynamically add levels
    if almost_conv then
      C.printf('almost converged \n')
      var rem = 0.0
      for i =-2, 1 do
        var lvl = NUM_LEVELS + i -1
        rem max= y_mean[lvl]*pow(2,alpha*i)/(pow(2,alpha)-1)
      end
      if rem > TOLERANCE/sqrt(2) then
        C.printf('add level \n')
        NUM_LEVELS += 1
        y_var[NUM_LEVELS-1] = y_var[NUM_LEVELS-2] / pow(2, beta)
        num_samples[NUM_LEVELS-1] = 0
       
        -- Update estimates for the optimal number of samples.
        for lvl = 0, NUM_LEVELS do
          y_costs[lvl] = pow(2, gamma*lvl) 
        end

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
      end
      C.printf('level added \n')
    end

    -- Print output.
    C.printf('Iteration %d:\n', iter)
    C.printf('  y_costs =')
    for lvl = 0, NUM_LEVELS do
      C.printf(' %e', y_costs[lvl])
    end
  
    C.printf('\n')
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