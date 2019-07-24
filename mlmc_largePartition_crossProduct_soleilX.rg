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
#include <string.h>
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
-------------------------------------------------------------------------------
-- Constants & inputs
-------------------------------------------------------------------------------

local NUM_LEVELS = 3
local NUM_UNCERTAINTIES = 1
local SEED = 1237
local MAX_SAMPLES_PER_LEVEL = 1000
local MAX_ITERS = 10
local TOLERANCE = 0.0001
local BATCH_SIZE = 100
local NUM_BATCHES = MAX_SAMPLES_PER_LEVEL / BATCH_SIZE

-- Enumeration of states that a sample can be in.
local State = {
  INACTIVE = 0,
  ACTIVE = 1,
  COMPLETED = 2,
}

local NUM_U_INPUT = 9000
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
  --var j = 0
  for s in samples do
    if s.state == State.ACTIVE then
      --__fence(__execution, __block)
      --C.printf('i_in_batch %d: active \n', j)
      --__fence(__execution, __block)
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
    else
      --__fence(__execution, __block)
      --C.printf('i_in_batch %d \n', j)
      --__fence(__execution, __block)
    end
    --j += 1
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

-- Main
-------------------------------------------------------------------------------

task main()

  --C.printf('%d \n', NUM_BATCHES)
  __fence(__execution, __block)
  var t_start = regentlib.c.legion_get_current_time_in_micros()
  __fence(__execution, __block)

  -- Initialize RNG.
  C.srand48(SEED)
  -- Inputs
  var opt_samples : int[NUM_LEVELS] = array(100,100,100)
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
  
  var index_space = ispace(int2d,{NUM_LEVELS,MAX_SAMPLES_PER_LEVEL})
  var samples = region(index_space, Sample)
  fill(samples.state, State.INACTIVE)
  
  --color each sample by level
  var coloring_level = regentlib.c.legion_domain_point_coloring_create()
  --i_level is the color
  for i_level = 0, NUM_LEVELS do
    var range = rect2d{int2d{i_level, 0}, int2d{i_level+1, MAX_SAMPLES_PER_LEVEL}}
    regentlib.c.legion_domain_point_coloring_color_domain(coloring_level, int1d(i_level), range)
  end
  var p_samples_by_level = partition(disjoint, samples, coloring_level, ispace(int1d,NUM_LEVELS))
  regentlib.c.legion_domain_point_coloring_destroy(coloring_level)

  --color each sample by batch
  var coloring_batch = regentlib.c.legion_domain_point_coloring_create()
  -- i_batch is the color
  for i_batch = 0, NUM_BATCHES do
    var lo = i_batch*BATCH_SIZE
    var hi = (i_batch+1)*BATCH_SIZE - 1
    var range = rect2d{int2d{0,lo}, int2d{NUM_LEVELS-1,hi}}
    regentlib.c.legion_domain_point_coloring_color_domain(coloring_batch, int1d(i_batch), range)
  end
  var p_samples_by_batch = partition(disjoint, samples, coloring_batch, ispace(int1d, NUM_BATCHES))
  regentlib.c.legion_domain_point_coloring_destroy(coloring_batch)

  var p_samples_fine = cross_product(p_samples_by_level, p_samples_by_batch)

  --var color_space_fine = ispace(int2d,{NUM_LEVELS,MAX_SAMPLES_PER_LEVEL/BATCH_SIZE})
  --var p_samples_fine = partition(equal, samples, color_space_fine)
  --var color_space_by_level = ispace(int2d,{NUM_LEVELS,1})
  --var p_samples_by_level = partition(equal, samples, color_space_by_level)

  -- Main loop 
  --If reading in uncertainties from file
  var f : &C.FILE
  f = C.fopen("uncertainties.txt", "rb")
  read_header(f)
  --regentlib.assert(read_header(f) == NUM_U_INPUT, "Input file wrong number of uncertainties.")
  var uncertainties : double[NUM_U_INPUT]
  for i=0, NUM_U_INPUT do
    var value : double[1]
    regentlib.assert(read_line(f, value), "Less data than expected")
    uncertainties[i]=value[0]
  end
  C.fclose(f)
  
  --for i = 0, 10 do
  --  C.printf("%f\n", uncertainties[i])
  --end

 -- var fp = C.fopen('rand_1000.csv', 1024, "r")
 -- var line : int8[1024]
 -- for i=0, 10 do
 --   C.fgets(line, 1024, fp)
 --   var index = C.atoi(C.strtok(line, ","))
 --   C.printf('%d \n', index)
 --   var mass_x = 0.0
 --   mass_x = C.atof(C.strtok([&int8](0), ","))
 --   C.printf('%d \n', i)
 --   C.printf('%f \n', mass_x)
 --   --C.printf(C.fgets(line, 1024, fp))
 --   --var u = C.fgets(line, fp)
 --   --C.pintf('%f \n', C.fgets(line, fp))
 -- end
 -- C.fclose(fp)

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
          --samples[{lvl,i}].uncertainties[j] = .5
          --samples[{lvl,i}].uncertainties[j] = C.drand48()
          samples[{lvl,i}].uncertainties[j] = uncertainties[lvl*MAX_SAMPLES_PER_LEVEL + i] --not set up for NUM_UNCERTAINTIES > 0 yet
        end
      end
      --want floor for i_batch_start, which is equivalent to casting to (int)
      --want ceil for i_end but ceil returns double so manually do ceil
      --(int) truncates and returns int
      --can't use floor() and ceil() b/c returns double so can't iterate over; 
      --if convert double with (int) get untyped which can't iterate over
      var i_batch_start = (int) ((double) (num_samples[lvl])/BATCH_SIZE)
      var i_batch_end = (int) ((double) (opt_samples[lvl])/BATCH_SIZE)
      if (double) (opt_samples[lvl])/BATCH_SIZE > i_batch_end then
        --C.printf('i_batch_end += 1 \n')
        i_batch_end += 1
      end
      --__fence(__execution, __block)
      --C.printf('num_samples[lvl] %d \n', num_samples[lvl])
      --C.printf('opt_samples[lvl] %d \n', opt_samples[lvl])
      --C.printf('num_samples[lvl]/BATCH_SIZE %f \n', (double) (num_samples[lvl])/BATCH_SIZE)
      --C.printf('opt_samples[lvl]/BATCH_SIZE %f \n', (double) (opt_samples[lvl])/BATCH_SIZE)
      --C.printf('i_batch_start %d \n', i_batch_start)
      --C.printf('i_batch_end %d \n', i_batch_end)
      --__fence(__execution, __block)
      for i_batch = i_batch_start, i_batch_end do
        --__fence(__execution, __block)
        --C.printf('i_batch: %d \n', i_batch)
        --__fence(__execution, __block)
        -- Invoke `eval_samples` for a set of samples containing just the
        -- newly created sample. This task will be launched asynchronously; the
        -- main task will continue running and launching new tasks. The runtime
        -- will analyze the dependencies between the newly launched task and
        -- all tasks launched before it. It will conclude that the new task is
        -- independent from all the others, and thus can be queued to run
        -- immediately.
        eval_samples(p_samples_fine[{lvl,i_batch}])
      end
      --__fence(__execution, __block)
      --C.printf('level: %d \n', lvl)
      --C.printf('pre num_samples[lvl]: %d \n', num_samples[lvl])
      num_samples[lvl] max= opt_samples[lvl]
      --C.printf('post num_samples[lvl]: %d \n', num_samples[lvl])
      --C.printf('pre opt_samples[lvl]: %d \n', opt_samples[lvl])
      --__fence(__execution, __block)
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
    -- Print output.
    --__fence(__execution, __block)
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
    --C.printf('  opt_samples =')
    --for lvl = 0, NUM_LEVELS do
    --  C.printf(' %d', opt_samples[lvl])
    --end
    --C.printf('\n')
    --__fence(__execution, __block)
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
  --print final results
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
