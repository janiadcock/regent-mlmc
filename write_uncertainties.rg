import 'regent'

local C = terralib.includecstring[[
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
]]

task toplevel()
  var out_file = C.fopen("uncertainties_large.txt", "w")
  var N = 100000
  C.fprintf(out_file, "%d\n",N)
  for i = 0,N do
    var u = C.drand48()
    C.fprintf(out_file, "%lf\n", u)
  end
  C.fclose(out_file)
end

regentlib.start(toplevel)
