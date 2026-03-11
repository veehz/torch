import { Tensor } from './tensor';
import { TorchFunction } from './functions/base';
import { Module } from './nn/base';
import { no_grad } from './grad_mode';
import { eventBus, events } from './util';

/**
 * A graph node in the exported program, matching PyTorch's FX node format.
 */
export interface GraphNode {
  /** Node type: 'placeholder' for inputs/params, 'call_function' for ops, 'output' for result */
  op: 'placeholder' | 'call_function' | 'output';
  /** Unique name for this node (e.g. "add", "linear_1") */
  name: string;
  /** Operation target (e.g. "aten.add.default") */
  target: string;
  /** References to input node names */
  args: (string | string[])[];
  /** Output tensor shape, if available */
  val_shape?: number[];
}

export interface InputSpec {
  kind: 'PARAMETER' | 'USER_INPUT';
  name: string;
  target?: string;
}

export interface OutputSpec {
  kind: 'USER_OUTPUT';
  name: string;
}

export interface GraphSignature {
  input_specs: InputSpec[];
  output_specs: OutputSpec[];
}

/**
 * Maps our internal op names to PyTorch's aten operator names.
 * See: https://docs.pytorch.org/docs/2.10/user_guide/torch_compiler/torch.compiler_ir.html
 */
export const _atenMap: Record<string, string> = {
  'add': 'aten.add.Tensor',
  'sub': 'aten.sub.Tensor',
  'mul': 'aten.mul.Tensor',
  'div': 'aten.div.Tensor',
  'pow': 'aten.pow.Tensor_Tensor',
  'powint': 'aten.pow.Tensor_Scalar',
  'fmod': 'aten.fmod.Tensor',
  'maximum': 'aten.maximum.default',
  'minimum': 'aten.minimum.default',
  'log': 'aten.log.default',
  'sqrt': 'aten.sqrt.default',
  'exp': 'aten.exp.default',
  'square': 'aten.square.default',
  'abs': 'aten.abs.default',
  'sign': 'aten.sign.default',
  'neg': 'aten.neg.default',
  'reciprocal': 'aten.reciprocal.default',
  'nan_to_num': 'aten.nan_to_num.default',
  'reshape': 'aten.reshape.default',
  'squeeze': 'aten.squeeze.dim',
  'unsqueeze': 'aten.unsqueeze.default',
  'expand': 'aten.expand.default',
  'sin': 'aten.sin.default',
  'cos': 'aten.cos.default',
  'tan': 'aten.tan.default',
  'sum': 'aten.sum.default',
  'mean': 'aten.mean.default',
  'min': 'aten.min.default',
  'max': 'aten.max.default',
  'transpose': 'aten.transpose.int',
  'matmul': 'aten.matmul.default',
  'relu': 'aten.relu.default',
  'sigmoid': 'aten.sigmoid.default',
  'lt': 'aten.lt.Tensor',
  'gt': 'aten.gt.Tensor',
  'le': 'aten.le.Tensor',
  'ge': 'aten.ge.Tensor',
  'eq': 'aten.eq.Tensor',
  'ne': 'aten.ne.Tensor',
  'conv1d': 'aten.conv1d.default',
  'conv2d': 'aten.conv2d.default',
  'conv3d': 'aten.conv3d.default',
  'linear': 'aten.linear.default',
};

/**
 * Maps our internal op names to PyTorch's aten operator names with default value.
 */
function toAtenTarget(opName: string): string {
  return _atenMap[opName] || `aten.${opName}.default`;
}

/**
 * Manages unique node name generation with PyTorch-style deduplication.
 * E.g. first "add" -> "add", second "add" -> "add_1"
 */
class NameGenerator {
  private counts = new Map<string, number>();

  generate(baseName: string): string {
    const count = this.counts.get(baseName) || 0;
    this.counts.set(baseName, count + 1);
    return count === 0 ? baseName : `${baseName}_${count}`;
  }
}

/**
 * An exported program, matching PyTorch's ExportedProgram structure.
 */
export class ExportedProgram {
  constructor(
    public graph: GraphNode[],
    public graph_signature: GraphSignature,
    public parameters: Map<string, { data: number[], shape: number[] }>
  ) { }

  toString(): string {
    const lines: string[] = ['ExportedProgram:'];

    // Format forward signature
    const inputArgs = this.graph
      .filter(n => n.op === 'placeholder')
      .map(n => {
        const shape = n.val_shape ? JSON.stringify(n.val_shape) : '?';
        return `${n.name}: "${shape}"`;
      })
      .join(', ');
    lines.push(`    class GraphModule(torch.nn.Module):`);
    lines.push(`        def forward(self, ${inputArgs}):`);

    // Operations
    for (const node of this.graph) {
      if (node.op === 'call_function') {
        const args = node.args.join(', ');
        lines.push(`            ${node.name} = ${node.target}(${args})`);
      } else if (node.op === 'output') {
        lines.push(`            return (${node.args.join(', ')},)`);
      }
    }

    lines.push('');
    lines.push('Graph signature:');
    lines.push('    # inputs');
    for (const spec of this.graph_signature.input_specs) {
      const target = spec.target ? ` target='${spec.target}'` : '';
      lines.push(`    ${spec.name}: ${spec.kind}${target}`);
    }
    lines.push('    # outputs');
    for (const spec of this.graph_signature.output_specs) {
      lines.push(`    ${spec.name}: ${spec.kind}`);
    }

    return lines.join('\n');
  }
}

/**
 * Export a module's forward pass as an ExportedProgram.
 *
 * This traces the module's forward() with the given sample inputs
 * and captures the computation graph. Similar to PyTorch's torch.export.export().
 *
 * Named `export_` to avoid conflict with the JavaScript `export` keyword.
 *
 * @param module The nn.Module to export
 * @param sampleInputs Sample input tensors for tracing
 * @returns An ExportedProgram containing the traced graph
 */
export function export_(
  module: Module,
  sampleInputs: Tensor[]
): ExportedProgram {
  const graph: GraphNode[] = [];
  const nameGen = new NameGenerator();

  // Map tensor IDs to their graph node names
  const tensorIdToName = new Map<number, string>();

  // 1. Create placeholder nodes for parameters
  const namedParams = module.named_parameters();
  const paramTensorIds = new Set<number>();
  const inputSpecs: InputSpec[] = [];

  for (const [paramPath, param] of namedParams) {
    // Convert "linear.weight" -> "p_linear_weight" (PyTorch convention)
    const placeholderName = 'p_' + paramPath.replace(/\./g, '_');
    const nodeName = nameGen.generate(placeholderName);
    tensorIdToName.set(param.id, nodeName);
    paramTensorIds.add(param.id);

    graph.push({
      op: 'placeholder',
      name: nodeName,
      target: nodeName,
      args: [],
      val_shape: param.shape,
    });

    inputSpecs.push({
      kind: 'PARAMETER',
      name: nodeName,
      target: paramPath,
    });
  }

  // 2. Create placeholder nodes for user inputs
  for (let i = 0; i < sampleInputs.length; i++) {
    const baseName = 'input';
    const nodeName = nameGen.generate(baseName);
    tensorIdToName.set(sampleInputs[i].id, nodeName);

    graph.push({
      op: 'placeholder',
      name: nodeName,
      target: nodeName,
      args: [],
      val_shape: sampleInputs[i].shape,
    });

    inputSpecs.push({
      kind: 'USER_INPUT',
      name: nodeName,
    });
  }

  // 3. Trace the forward pass, recording operations
  const handler = (e: CustomEvent) => {
    const { operation, args, result } = e.detail as {
      operation: TorchFunction;
      args: (Tensor | number | number[] | boolean)[];
      result: Tensor;
    };

    const opName = operation.opName;
    if (!opName) return; // Skip if no opName (shouldn't happen)

    // Build arg references
    const nodeArgs: (string | string[])[] = [];
    for (const arg of args) {
      if (arg instanceof Tensor) {
        const name = tensorIdToName.get(arg.id);
        if (name) {
          nodeArgs.push(name);
        }
        // If not found, it's an intermediate constant — skip
      }
      // Numbers and arrays are non-tensor args; we don't include them
      // in the graph node args to match PyTorch's behavior for simple cases
    }

    // Generate node name from opName
    const nodeName = nameGen.generate(opName);
    tensorIdToName.set(result.id, nodeName);

    graph.push({
      op: 'call_function',
      name: nodeName,
      target: toAtenTarget(opName),
      args: nodeArgs,
      val_shape: result.shape,
    });
  };

  eventBus.addEventListener(
    events.OPERATION_AFTER_FORWARD,
    handler as EventListener
  );

  let output: Tensor;
  try {
    output = no_grad(() => module.forward(...sampleInputs));
  } finally {
    eventBus.removeEventListener(
      events.OPERATION_AFTER_FORWARD,
      handler as EventListener
    );
  }

  // 4. Add output node
  const outputName = tensorIdToName.get(output.id) || 'output';
  graph.push({
    op: 'output',
    name: 'output',
    target: 'output',
    args: [outputName],
  });

  const outputSpecs: OutputSpec[] = [{
    kind: 'USER_OUTPUT',
    name: outputName,
  }];

  // 5. Collect parameters
  const parameters = new Map<string, { data: number[], shape: number[] }>();
  for (const [paramPath, param] of namedParams) {
    parameters.set(paramPath, {
      data: [...param.data],
      shape: [...param.shape],
    });
  }

  return new ExportedProgram(
    graph,
    { input_specs: inputSpecs, output_specs: outputSpecs },
    parameters
  );
}
