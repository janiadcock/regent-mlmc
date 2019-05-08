import "regent"
local c = regentlib.c
local cstring = terralib.includec("string.h")

local N = 9000

terra print_usage_and_abort()
  c.printf("Usage: regent.py io_part.rg [OPTIONS]\n")
  c.printf("OPTIONS\n")
  c.printf("  -h            : Print the usage and exit.\n")
  c.printf("  -i {file}     : Use {file} as input.\n")
  c.exit(0)
end

struct RV_Config
{
  file_name      : int8[512],
  input_size     : uint64;
}

terra RV_Config:initialize_from_command()
  var args = c.legion_runtime_get_input_args()
  var i = 1
  var input_given = false
  while i < args.argc do
    if cstring.strcmp(args.argv[i], "-h") == 0 then
      print_usage_and_abort()
    elseif cstring.strcmp(args.argv[i], "-i") == 0 then
      i = i + 1
      var file = c.fopen(args.argv[i], "rb")
      if file == nil then
        c.printf("File '%s' doesn't exist!\n", args.argv[i])
        c.abort()
      end
      cstring.strcpy(self.file_name, args.argv[i])
      c.fscanf(file, "%llu\n", &self.input_size)
      input_given = true
      c.fclose(file)
    end
    i = i + 1
  end
  if not input_given then
    c.printf("Input file must be given!\n\n")
    print_usage_and_abort()
  end
end

terra skip_header(f : &c.FILE)
  var x: uint64
  c.fscanf(f, "%llu\n", &x)
end

terra read_line(f : &c.FILE, value : &double)
  return c.fscanf(f, "%lf\n", &value[0]) == 1
end

task toplevel()
  c.printf("START \n")
  var config : RV_Config
  config:initialize_from_command()

  c.printf("**********************************\n")
  c.printf("* Partitioning and IO Example    *\n")
  c.printf("*                                *\n")
  c.printf("* Number of Random Variables imported       : %6lu\n",  config.input_size)
  c.printf("**********************************\n")

  --regentlib.assert(N == config.input_size, "Input file wrong number of uncertainties.")
  var uncertainties : double[N]
  var f = c.fopen("uncertainties.txt", "rb")
  --var f = c.fopen(config.file_name, "rb")

  skip_header(f) 
  for i = 0, N do 
    var value : double[1]
    regentlib.assert(read_line(f, value), "Less data than expected")
    uncertainties[i] = value[0]
  end
  c.fclose(f)

  --for i=0,N do
  --  c.printf("%f \n", uncertainties[i]) 
  --end
end

regentlib.start(toplevel)
