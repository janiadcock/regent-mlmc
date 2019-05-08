import "regent"
local c = regentlib.c
local cstring = terralib.includec("string.h")

local N = 9000

terra read_header(f : &c.FILE)
  var x: uint64
  return c.fscanf(f, "%llu\n", &x)
end

terra read_line(f : &c.FILE, value : &double)
  return c.fscanf(f, "%lf\n", &value[0]) == 1
end


task toplevel()
  var f : &c.FILE
  f = c.fopen("uncertainties.txt", "rb")

  read_header(f)
  --regentlib.assert(read_header(f) == N, "Input file wrong number of uncertainties.")

  var uncertainties : double[N]
  for i=0,N do
    var value : double[1]
    regentlib.assert(read_line(f, value), "Less data than expected")
    --regentlib.assert(c.fscanf(f, "%lf\n", &value[0])==1, "Less data than expected")
    uncertainties[i] = value[0]
  end
  c.fclose(f)

  for i =0,N do
    c.printf("%f\n", uncertainties[i])
  end
end

regentlib.start(toplevel)
