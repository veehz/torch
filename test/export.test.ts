import { assert } from 'chai';
import { _atenMap } from '../src/export';
import { _getAllOperationNames } from '../src/functions/registry';

// Import torch to trigger operation registration side effects
import '@sourceacademy/torch';

describe('Export', () => {
  // List from https://docs.pytorch.org/docs/2.10/user_guide/torch_compiler/torch.compiler_ir.html

  const EXCLUDED_OPERATIONS = new Set([
    "__left_index__",
    "__right_index__",
  ]);

  it('has all supported operations', () => {
    for (const opName of _getAllOperationNames()) {
      assert.isTrue(opName in _atenMap || EXCLUDED_OPERATIONS.has(opName),
        `Missing aten mapping for operation: ${opName}. ` +
        `Please add it in export.ts or exclude it in export.test.ts`);
    }
  });
});
