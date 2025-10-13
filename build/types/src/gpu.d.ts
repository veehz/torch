import { GPU, Texture } from '@veehz/gpu.js';
export { Texture };
/**
 * Problems with gpu:
 * - source array too large when all the options dynamicoutput dynamicarguments pipeline immutable are true
 * - if pipeline is not used, then the memory transfer is too slow
 * - if dynamicarguments is not used, then the source array is too large error will appear (https://github.com/gpujs/gpu.js/issues/495)
 * - if dynamicoutputs is not used, then loops and broadcasting cannot be used
 *
 * CPU is currently fast enough as well.
 */
declare const gpu: GPU;
export default gpu;
export { GPU };
