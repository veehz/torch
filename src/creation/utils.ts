export function get_shape_from_args(args: number[] | number[][]): number[] {
  if (Array.isArray(args[0])) {
    return args[0];
  }

  return args as number[];
}
