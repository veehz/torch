// unseeded by default
let _rng = () => Math.random();

export function getRng(): () => number {
  return _rng;
}

export function manual_seed(seed: number): number {
  seed = seed >>> 0; // to uint32
  _rng = mulberry32(seed);
  return seed;
}

export function seed(): number {
  const s = (Math.random() * 0xffffffff) >>> 0;
  _rng = mulberry32(s);
  return s;
}

// https://stackoverflow.com/a/47593316
function mulberry32(seed: number): () => number {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function uniformDist(min = 0, max = 1) {
  return () => min + getRng()() * (max - min);
}

// https://stackoverflow.com/a/36481059
export function normalDist(mean = 0, std = 1) {
  return function () {
    const u = 1 - getRng()(); // [0,1) -> (0,1]
    const v = getRng()();
    const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    return z * std + mean;
  };
}
