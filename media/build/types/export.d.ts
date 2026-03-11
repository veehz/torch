import { Tensor } from './tensor';
import { Module } from './nn/base';
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
export declare const _atenMap: Record<string, string>;
/**
 * An exported program, matching PyTorch's ExportedProgram structure.
 */
export declare class ExportedProgram {
    graph: GraphNode[];
    graph_signature: GraphSignature;
    parameters: Map<string, {
        data: number[];
        shape: number[];
    }>;
    constructor(graph: GraphNode[], graph_signature: GraphSignature, parameters: Map<string, {
        data: number[];
        shape: number[];
    }>);
    toString(): string;
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
export declare function export_(module: Module, sampleInputs: Tensor[]): ExportedProgram;
