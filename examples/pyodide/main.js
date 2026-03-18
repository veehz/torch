import { loadPyodide } from 'pyodide';
import * as torch from 'torch';
import { readFileSync, writeFileSync, mkdirSync, readdirSync, existsSync } from 'fs';
import { execSync } from 'child_process';
import path from 'path';

const PY_DIR = './py';
const CACHE_DIR = './.cache';

function getPyFiles() {
  return readdirSync(PY_DIR).filter(f => f.endsWith('.py')).sort();
}

async function setupPyodide() {
  const pyodide = await loadPyodide();
  pyodide.globals.set('js_torch', torch);
  pyodide.globals.set('_js_is_null', (x) => x == null);
  pyodide.runPython(readFileSync('./bridge.py', 'utf8'));
  return pyodide;
}

async function runWithPyodide(pyodide, code) {
  // Reset grad state to enabled before each file.
  // disable_no_grad(prev) restores a previous state; passing True enables grad.
  pyodide.runPython('js_torch.disable_no_grad(True)');
  pyodide.globals.set('_test_code', code);
  pyodide.runPython(`
import sys, builtins
from io import StringIO
_ns = {'torch': torch, '__builtins__': builtins}
_saved = sys.stdout
sys.stdout = StringIO()
try:
    exec(_test_code, _ns)
    _output = sys.stdout.getvalue()
finally:
    sys.stdout = _saved
`);
  return pyodide.globals.get('_output');
}

async function cmdGen() {
  if (!existsSync(CACHE_DIR)) mkdirSync(CACHE_DIR);
  const files = getPyFiles();
  let ok = 0, fail = 0;
  for (const file of files) {
    const name = path.basename(file, '.py');
    try {
      const output = execSync(
        `python3 -c "import torch, builtins; exec(open('${PY_DIR}/${file}').read(), {'torch': torch, '__builtins__': builtins})"`,
        { encoding: 'utf8' }
      );
      writeFileSync(path.join(CACHE_DIR, `${name}.out`), output);
      console.log(`✓ ${name}`);
      ok++;
    } catch (e) {
      console.error(`✗ ${name}: ${e.stderr?.trim() ?? e.message}`);
      fail++;
    }
  }
  console.log(`\n${ok} generated, ${fail} failed`);
  if (fail > 0) process.exit(1);
}

async function cmdTest() {
  const files = getPyFiles();
  const pyodide = await setupPyodide();
  let passed = 0, failed = 0;

  for (const file of files) {
    const name = path.basename(file, '.py');
    const cachePath = path.join(CACHE_DIR, `${name}.out`);
    if (!existsSync(cachePath)) {
      console.log(`✗ ${name}: no cache (run 'yarn gen-cache' first)`);
      failed++;
      continue;
    }

    const code = readFileSync(path.join(PY_DIR, file), 'utf8');
    let actual;
    try {
      actual = await runWithPyodide(pyodide, code);
    } catch (e) {
      console.log(`✗ ${name}: ${e.message}`);
      failed++;
      continue;
    }

    const expected = readFileSync(cachePath, 'utf8');
    const aMarked = actual.split('\n').filter(l => l.startsWith('>'));
    const eMarked = expected.split('\n').filter(l => l.startsWith('>'));
    const mismatches = [];
    const maxLen = Math.max(aMarked.length, eMarked.length);
    for (let i = 0; i < maxLen; i++) {
      if (aMarked[i] !== eMarked[i]) {
        mismatches.push(`  [assertion ${i + 1}] expected: ${JSON.stringify(eMarked[i] ?? '(missing)')}`);
        mismatches.push(`  [assertion ${i + 1}]   actual: ${JSON.stringify(aMarked[i] ?? '(missing)')}`);
      }
    }
    if (mismatches.length === 0) {
      console.log(`✓ ${name} (${aMarked.length} assertions)`);
      passed++;
    } else {
      console.log(`✗ ${name}: ${mismatches.length / 2} assertion(s) failed`);
      mismatches.forEach(l => console.log(l));
      failed++;
    }
  }

  console.log(`\n${passed} passed, ${failed} failed`);
  if (failed > 0) process.exit(1);
}

const [mode] = process.argv.slice(2);
if (mode === 'gen') cmdGen().catch(e => { console.error(e); process.exit(1); });
else if (mode === 'test') cmdTest().catch(e => { console.error(e); process.exit(1); });
else {
  console.error('Usage: node main.js [gen|test]');
  process.exit(1);
}
