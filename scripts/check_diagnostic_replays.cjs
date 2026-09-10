// Validate real exports using the exact parser used by the replay page.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
assert(process.env.SLURM_JOB_ID, 'Run validation in a compute allocation');
const root = path.resolve(process.argv[2]);
const parserPath = path.resolve('ui/shared/replay-format.js');
const parser = require(parserPath);
const reasons = {
  'Checkmate': 'checkmate',
  'Stalemate (loss for stalemated side)': 'stalemate',
  'Threefold repetition': 'repetition',
  'Max plies reached': 'move_limit',
  'Chess king captured': 'royal_captured',
  'Xiangqi general captured': 'royal_captured',
};
const groups = [
  ['baseline', 'run-r01/arena/games', 120],
  ['sampling', 'run-r01/after/games', 48],
  ['teacher', 'evidence-final/replays', 3],
  ['mate', 'confirm-r05/replays', 168],
];
const summary = [];
for (const [label, relative, expected] of groups) {
  const source = path.join(root, relative);
  const names = fs.readdirSync(source).filter(name => name.endsWith('.json')).sort();
  assert.equal(names.length, expected);
  const exported = [];
  let boards = 0, moves = 0;
  const out = path.join(root, 'ui-replays', label);
  fs.mkdirSync(out, {recursive: true});
  for (const name of names) {
    const original = JSON.parse(fs.readFileSync(path.join(source, name), 'utf8'));
    const isTerminal = Boolean(reasons[original.reason]);
    const result = original.winner ? original.winner + '_win' : isTerminal ? 'draw' : 'ongoing';
    const row = {...original, result_code: result,
      reason_code: reasons[original.reason] || null,
      meta: {...original.meta, reason: original.reason},
      diagnostic_origin: relative + '/' + name};
    const text = JSON.stringify(row);
    assert.equal(parser.parse(text).length, 1);
    assert.equal(row.states_ascii.length, row.moves.length + 1);
    for (const move of row.moves) assert(parser.parseMove(move));
    fs.writeFileSync(path.join(out, name), text + '\n');
    exported.push(row);
    boards += row.states_ascii.length;
    moves += row.moves.length;
  }
  assert.equal(parser.parse(exported.map(row => JSON.stringify(row)).join('\n')).length, expected);
  summary.push({group: label, recordings: expected, boards, moves, json_and_jsonl: true});
}
const result = {complete: true, visual_browser_acceptance: false,
  parser_sha256: crypto.createHash('sha256').update(fs.readFileSync(parserPath)).digest('hex'),
  job_id: process.env.SLURM_JOB_ID, recordings: summary};
fs.writeFileSync(path.join(root, 'replay-import-check.json'), JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify(result));
