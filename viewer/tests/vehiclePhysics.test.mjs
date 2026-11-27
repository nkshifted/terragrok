import test from 'node:test';
import assert from 'node:assert/strict';

import { VEHICLES, createVehicleState, isWater, stepVehicle, summonSpot } from '../src/vehiclePhysics.js';

/** Terrain from a height function; nothing blocks movement unless `wallX` is set. */
function world(height, { lake = () => 0, river = () => 0, wallX = Infinity } = {}) {
  return {
    height,
    lake,
    river,
    move: (x, z, nx, nz) => (nx > wallX ? { x, z, blocked: true } : { x: nx, z: nz, blocked: false }),
  };
}

const FLAT = world(() => 10);
// Dry land for x < 0, sea deepening to the east
const COAST = world((x) => (x < 0 ? 5 : -Math.min(30, x * 0.5) - 0.3));
const DT = 1 / 60;

function run(s, terrain, input, seconds) {
  for (let i = 0; i < seconds / DT; i++) stepVehicle(s, { throttle: 0, steer: 0, boost: false, action: false, ...input }, DT, terrain);
  return s;
}

const EAST = Math.PI / 2; // heading is atan2(dx, dz): +x

test('every vehicle has a distinct summon key', () => {
  const keys = Object.values(VEHICLES).map((v) => v.key);
  assert.equal(new Set(keys).size, keys.length);
});

test('a jeep accelerates to its top speed in a straight line', () => {
  const s = run(createVehicleState('jeep', 0, 0, EAST, FLAT), FLAT, { throttle: 1 }, 6);
  assert.ok(Math.abs(s.speed - VEHICLES.jeep.maxSpeed) < 0.5);
  assert.ok(s.x > 400 && Math.abs(s.z) < 1e-6);
});

test('steering right turns clockwise seen from above (heading decreases)', () => {
  const s = run(createVehicleState('horse', 0, 0, 0, FLAT), FLAT, { throttle: 1, steer: 1 }, 0.5);
  assert.ok(s.heading < 0);
  // Heading 0 is +z (south); a right turn from there bends towards -x (west)
  assert.ok(s.x < 0);
});

test('a jeep cannot turn on the spot but a horse can', () => {
  const jeep = run(createVehicleState('jeep', 0, 0, 0, FLAT), FLAT, { steer: 1 }, 1);
  const horse = run(createVehicleState('horse', 0, 0, 0, FLAT), FLAT, { steer: 1 }, 1);
  assert.equal(jeep.heading, 0);
  assert.ok(horse.heading < -1);
});

const ramp = (deg) => world((x) => Math.max(0, x) * Math.tan((deg * Math.PI) / 180) + 5);

test('a very steep slope stops a jeep that a horse still climbs', () => {
  const hill = ramp(56);
  const jeep = run(createVehicleState('jeep', -50, 0, EAST, hill), hill, { throttle: 1 }, 6);
  const horse = run(createVehicleState('horse', -50, 0, EAST, hill), hill, { throttle: 1 }, 6);
  assert.ok(jeep.x < 12 && jeep.blocked === 'terrain', `jeep reached x=${jeep.x}`);
  assert.ok(horse.x > 60, `horse reached x=${horse.x}`);
  assert.ok(horse.pitch > 0.7, 'the horse is nose-up on the climb');
});

test('both climb a slope a person would scramble up, and neither climbs a wall', () => {
  const steep = ramp(45);
  assert.ok(run(createVehicleState('jeep', -50, 0, EAST, steep), steep, { throttle: 1 }, 6).x > 60);
  const wall = ramp(75);
  for (const kind of ['jeep', 'horse']) {
    const s = run(createVehicleState(kind, -50, 0, EAST, wall), wall, { throttle: 1 }, 6);
    assert.ok(s.x < 12 && s.blocked === 'terrain', `${kind} reached x=${s.x}`);
  }
});

test('climbing costs speed', () => {
  const flat = run(createVehicleState('jeep', 0, 0, EAST, FLAT), FLAT, { throttle: 1 }, 8);
  const hill = ramp(40);
  const up = run(createVehicleState('jeep', 20, 0, EAST, hill), hill, { throttle: 1 }, 8);
  assert.ok(up.speed < flat.speed * 0.7 && up.speed > flat.speed * 0.4, `uphill ${up.speed} vs flat ${flat.speed}`);
});

test('a horse and a jeep jump, a boat does not', () => {
  for (const kind of ['horse', 'jeep']) {
    const s = run(createVehicleState(kind, 0, 0, EAST, FLAT), FLAT, { throttle: 1 }, 2);
    const x0 = s.x;
    stepVehicle(s, { throttle: 1, steer: 0, boost: false, action: true }, DT, FLAT);
    assert.ok(s.air, `${kind} leaves the ground`);
    let peak = s.y;
    let t = 0;
    while (s.air && t < 5) {
      stepVehicle(s, { throttle: 1, steer: 0, boost: false, action: false }, DT, FLAT);
      peak = Math.max(peak, s.y);
      t += DT;
    }
    assert.ok(peak - 10 > 3 && peak - 10 < 12, `${kind} rose ${peak - 10}`);
    assert.equal(s.y, 10, 'and lands back on the ground');
    assert.ok(s.x - x0 > 20, `${kind} carried ${s.x - x0} forward`);
  }
  const boat = createVehicleState('boat', 80, 0, EAST, COAST);
  stepVehicle(boat, { throttle: 1, steer: 0, boost: false, action: true }, DT, COAST);
  assert.equal(boat.air, false);
});

test('a ramp taken at speed throws the vehicle across a gap', () => {
  // Up a 25° ramp to x = 80, then a 40-wide trench, then level ground at the ramp-top height
  const top = 80 * Math.tan((25 * Math.PI) / 180) + 100;
  const gap = world((x) => (x < 0 ? 100 : x <= 80 ? x * Math.tan((25 * Math.PI) / 180) + 100 : x < 120 ? top - 60 : top));
  const fast = createVehicleState('jeep', -260, 0, EAST, gap);
  let flew = false;
  let lowest = Infinity;
  for (let i = 0; i < 8 / DT && fast.x < 140; i++) {
    stepVehicle(fast, { throttle: 1, steer: 0, boost: false, action: false }, DT, gap);
    flew ||= fast.air;
    if (fast.x > 80 && fast.x < 120) lowest = Math.min(lowest, fast.y);
  }
  assert.ok(flew, 'it left the ground at the lip');
  assert.ok(fast.x >= 120 && lowest > top - 20, `cleared the trench (x=${fast.x}, lowest ${lowest} vs lip ${top})`);
  // Crawling over the same lip just drops in
  const slow = createVehicleState('jeep', 70, 0, EAST, gap);
  for (let i = 0; i < 6 / DT; i++) stepVehicle(slow, { throttle: slow.speed < 12 ? 1 : 0, steer: 0, boost: false, action: false }, DT, gap);
  assert.ok(slow.x < 120 && slow.y < top - 40, `the slow one fell in (x=${slow.x}, y=${slow.y})`);
});

test('rolling ground does not make a vehicle hop', () => {
  const rolling = world((x) => 20 + 6 * Math.sin(x / 40));
  const s = createVehicleState('jeep', 0, 0, EAST, rolling);
  let hops = 0;
  for (let i = 0; i < 10 / DT; i++) {
    stepVehicle(s, { throttle: 1, steer: 0, boost: false, action: false }, DT, rolling);
    if (s.air) hops++;
  }
  assert.equal(hops, 0);
});

test('land vehicles stop at deep water and a boat stops at the shore', () => {
  const jeep = run(createVehicleState('jeep', -60, 0, EAST, COAST), COAST, { throttle: 1 }, 6);
  assert.ok(jeep.x < 3 && jeep.blocked === 'terrain');
  const boat = run(createVehicleState('boat', 80, 0, -EAST, COAST), COAST, { throttle: 1 }, 8);
  assert.ok(boat.x >= 0 && boat.x < 3 && boat.blocked === 'terrain', `boat reached x=${boat.x}`);
  assert.equal(boat.y, 0);
  assert.ok(isWater(COAST, boat.x, boat.z));
});

test('summoning respects the medium', () => {
  assert.equal(summonSpot('boat', FLAT, 0, 0).ok, false);
  const afloat = summonSpot('boat', COAST, -40, 0);
  assert.ok(afloat.ok && afloat.x >= 0, 'the boat appears on the nearest water');
  const ashore = summonSpot('jeep', COAST, 40, 0);
  assert.ok(ashore.ok && ashore.x < 0, 'the jeep appears on the nearest land');
  assert.equal(summonSpot('horse', COAST, 900, 0).ok, false);
  assert.equal(summonSpot('glider', COAST, 40, 0).ok, false);
  assert.equal(summonSpot('glider', FLAT, 0, 0).ok, true);
});

test('a lake floats a boat above its bed', () => {
  const lake = world(() => 40, { lake: (x) => (Math.abs(x) < 50 ? 1 : 0) });
  const s = createVehicleState('boat', 0, 0, EAST, lake);
  assert.ok(s.y > 40);
  run(s, lake, { throttle: 1 }, 6);
  assert.ok(s.x < 50, 'and keeps it inside the lake');
});

test('a canoe follows a river that a boat cannot enter', () => {
  // A river along z = 0 on dry land, 16 wide, running east into a lake at x > 200
  const land = world(() => 30, { river: (x, z) => (Math.abs(z) < 8 ? 1 : 0), lake: (x) => (x > 200 ? 1 : 0) });
  assert.ok(isWater(land, 0, 0, true) && !isWater(land, 0, 0, false));
  assert.equal(summonSpot('boat', land, 0, 0).ok, false);
  const put = summonSpot('canoe', land, 0, 40);
  assert.ok(put.ok && Math.abs(put.z) < 8, 'the canoe appears on the river');
  const canoe = run(createVehicleState('canoe', 0, 0, EAST, land), land, { throttle: 1 }, 8);
  assert.ok(canoe.x > 250, 'downriver and out onto the lake');
  const ashore = run(createVehicleState('canoe', 0, 0, 0, land), land, { throttle: 1 }, 3);
  assert.ok(Math.abs(ashore.z) < 8.5 && ashore.blocked === 'terrain', 'but not up the bank');
  const boat = run(createVehicleState('boat', 300, 0, -EAST, land), land, { throttle: 1 }, 8);
  assert.ok(boat.x > 199, 'the boat stays on the lake');
});

test('holding the action key lifts a glider, up to its ceiling', () => {
  const s = createVehicleState('glider', 0, 0, EAST, FLAT);
  run(s, FLAT, {}, 3);
  const y0 = s.y;
  run(s, FLAT, { action: true }, 10);
  assert.ok(s.lifting && s.y > y0 + 80, `climbed ${s.y - y0}`);
  run(s, FLAT, { action: true }, 120);
  assert.ok(s.y - 10 < VEHICLES.glider.ceiling + 40, `stopped near the ceiling at ${s.y - 10}`);
  assert.ok(!s.landed);
});

test('a glider climbs off the launch, glides down and lands', () => {
  const s = createVehicleState('glider', 0, 0, EAST, FLAT);
  let peak = s.y;
  let t = 0;
  while (!s.landed && t < 120) {
    stepVehicle(s, { throttle: 0, steer: 0, boost: false, action: false }, DT, FLAT);
    peak = Math.max(peak, s.y);
    t += DT;
  }
  assert.ok(peak > 10 + 15, `launch gained ${peak - 10} m`);
  assert.ok(s.landed, 'it comes down by itself');
  assert.ok(s.x > 200, `it covered ${s.x} m`);
});

test('diving a glider is faster and steeper than flaring', () => {
  const dive = run(createVehicleState('glider', 0, 0, EAST, world(() => -5000)), world(() => -5000), { throttle: 1 }, 8);
  const flare = run(createVehicleState('glider', 0, 0, EAST, world(() => -5000)), world(() => -5000), { throttle: -1 }, 8);
  assert.ok(dive.speed > flare.speed * 2);
  assert.ok(dive.vy < flare.vy - 10);
  assert.ok(dive.pitch < flare.pitch, 'nose lower in the dive');
});

test('the world edge stops a vehicle without teleporting it', () => {
  const edge = world(() => 10, { wallX: 30 });
  const s = run(createVehicleState('horse', 0, 0, EAST, edge), edge, { throttle: 1 }, 4);
  assert.ok(s.x <= 30 && s.x > 25 && s.blocked === 'edge');
});
