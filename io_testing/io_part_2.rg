import "regent"

local c = regentlib.c
local cstring = terralib.includec("string.h")

fspace FieldPart {
  value : double;
  color : int1d;
}

task print_field_part(reg : region(ispace(int1d), FieldPart))
where
  reads(reg.value, reg.color)
do
  -- Print out the color that legion thinks this subregion belongs to
  -- __runtime() gets the legion runtime object, and --raw gets the legion Logical Region object
  c.printf("Points and values for partition color %d\n", c.legion_logical_region_get_color_domain_point(__runtime(), __raw(reg)))
  for p in reg do
    c.printf("Point %d has value %lf and color %d\n", p, p.value, p.color)
  end
  c.printf("\n")
end

terra skip_header(f : &c.FILE)
  var x : uint64, y : uint64
  c.fscanf(f, "%llu\n%llu\n", &x, &y)
end

terra read_line(f : &c.FILE, value_and_color : &double)
  return c.fscanf(f, "%lf %lf\n", &value_and_color[0], &value_and_color[1]) == 2
end

task initialize_input(r_field_part   : region(ispace(int1d), FieldPart),
                      filename  : int8[512])
where
  reads writes(r_field_part)
do
  var f = c.fopen(filename, "rb")
  skip_header(f)
  var value_and_color : double[2]
  for p in r_field_part do
    regentlib.assert(read_line(f, value_and_color), "Less data that it should be")
    r_field_part[p].value = value_and_color[0]
    r_field_part[p].color = value_and_color[1]
  end
  c.fclose(f)
end

terra print_usage_and_abort()
  c.printf("Usage: regent.py partition_io_example.rg [OPTIONS]\n")
  c.printf("OPTIONS\n")
  c.printf("  -h            : Print the usage and exit.\n")
  c.printf("  -i {file}     : Use {file} as input.\n")
  c.exit(0)
end

struct ExampleConfig
{
  file_name      : int8[512],
  input_size     : uint64;
  num_partitions : uint64;
}

terra ExampleConfig:initialize_from_command()
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
      c.fscanf(file, "%llu\n%llu\n", &self.input_size, &self.num_partitions)
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

task toplevel()
  var config : ExampleConfig
  config:initialize_from_command()

  c.printf("**********************************\n")
  c.printf("* Partitioning and IO Example    *\n")
  c.printf("*                                *\n")
  c.printf("* Number of Elems       : %6lu *\n",  config.input_size)
  c.printf("* Number of Partitions  : %6lu *\n",  config.num_partitions)
  c.printf("**********************************\n")

  var r_field_part = region(ispace(int1d, config.input_size), FieldPart)
  initialize_input(r_field_part, config.file_name)

  var color_space = ispace(int1d, config.num_partitions)

  -- Passing a field as the first argument to partition tells Regent
  -- to partition the region by this field
  var p_field_part = partition(r_field_part.color, color_space)

  -- First we print the entire region, then look at each subregion
  -- You should see that the subregions are colored as specified in the file
  print_field_part(r_field_part)
  for c in color_space do
    print_field_part(p_field_part[c])
  end
end

regentlib.start(toplevel)
