-- Copyright 2019 Stanford University
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

import "regent"

local c = regentlib.c

local C = terralib.includecstring[[
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
]]

task f() : int
  var r = region(ispace(int2d, {3, 2}), int)
  var colors = ispace(int2d, {2, 2})

  var rc = c.legion_domain_point_coloring_create()
  c.legion_domain_point_coloring_color_domain(rc, int2d{0, 0}, rect2d{{0, 0}, {1, 1}})
  c.legion_domain_point_coloring_color_domain(rc, int2d{1,0}, rect2d{{2,0}, {2,1}})
  var p = partition(disjoint, r, rc, colors)
  c.legion_domain_point_coloring_destroy(rc)
  var r0 = p[{0, 0}]
  var r01 = p[{0,1}]
  var r10 = p[{1,0}]
  var r11 = p[{1,1}]
  --throws error that r21 an invalid partition, but r01, r10, and r11 don't so those are valid partitions
  --var r21 = p[{2,1}]
  fill(r, 1)
  fill(r0, 10)
  --sum equals 42 so partitions r01, r10, r11 are empty
  fill(r01, 100) 
  fill(r10, 1000)
  fill(r11, 10000)

  var t = 0
  for i in r do
    t += r[i]
  end
  return t
end

task main()
  regentlib.assert(f() == 2040, "test failed")
end
regentlib.start(main)
