// Script to check if GPU is supported on the system
import { GPU } from "../build/node/torch.node.es.js";

console.log("Is GPU supported:", GPU.isGPUSupported);
console.log("Is kernel map supported:", GPU.isKernelMapSupported);
console.log("Is off-screen canvas supported:", GPU.isOffscreenCanvasSupported);
console.log("Is WebGL supported:", GPU.isWebGLSupported);
console.log("Is WebGL2 supported:", GPU.isWebGL2Supported);
console.log("Is HeadlessGL supported:", GPU.isHeadlessGLSupported);
console.log("Is Canvas supported:", GPU.isCanvasSupported);
console.log("Is GPUHTMLImageArray supported:", GPU.isGPUHTMLImageArraySupported);
console.log("Is SinglePrecision supported:", GPU.isSinglePrecisionSupported);
