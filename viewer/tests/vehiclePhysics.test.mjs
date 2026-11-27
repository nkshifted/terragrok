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

// A beach: dry land sloping gently under the sea to the east
const BEACH = world((x) => 3 - Math.max(0, x + 60) * 0.05);

test('land vehicles stop where a beach gets too deep, and a boat stops at the shore', () => {
  const jeep = run(createVehicleState('jeep', -60, 0, EAST, BEACH), BEACH, { throttle: 1 }, 8);
  assert.ok(BEACH.height(jeep.x) >= -VEHICLES.jeep.wadeDepth - 0.2 && jeep.blocked === 'terrain' && !jeep.swamped, `jeep at x=${jeep.x}`);
  const horse = run(createVehicleState('horse', -60, 0, EAST, BEACH), BEACH, { throttle: 1 }, 8);
  assert.ok(horse.x > jeep.x + 5 && !horse.swamped, 'a horse wades further out');
  const boat = run(createVehicleState('boat', 80, 0, -EAST, COAST), COAST, { throttle: 1 }, 8);
  assert.ok(boat.x >= 0 && boat.x < 3 && boat.blocked === 'terrain', `boat reached x=${boat.x}`);
  assert.equal(boat.y, 0);
  assert.ok(isWater(COAST, boat.x, boat.z));
});

test('driving off a cliff into the sea is an arc and a splash, not a wall', () => {
  // A 60-high cliff at x = 0 with deep sea below it
  const cliff = world((x) => (x < 0 ? 60 : -25));
  for (const kind of ['jeep', 'horse']) {
    const s = createVehicleState(kind, -150, 0, EAST, cliff);
    let flew = false;
    let stalled = false;
    let t = 0;
    while (!s.swamped && t < 15) {
      stepVehicle(s, { throttle: 1, steer: 0, boost: true, action: false }, DT, cliff);
      flew ||= s.air;
      stalled ||= s.air && s.speed === 0;
      for (const k of ['x', 'y', 'z', 'vy', 'speed', 'pitch']) assert.ok(Number.isFinite(s[k]), `${kind} ${k} is ${s[k]}`);
      t += DT;
    }
    assert.ok(flew && !stalled, `${kind} keeps moving through the air`);
    assert.ok(s.swamped && s.x > 30, `${kind} came down out at sea (x=${s.x})`);
    assert.ok(s.y <= 0 && s.y >= -VEHICLES[kind].wadeDepth, `${kind} ends at the surface (y=${s.y})`);
  }
});

test('a frame with no elapsed time changes nothing', () => {
  const s = run(createVehicleState('jeep', 0, 0, EAST, FLAT), FLAT, { throttle: 1 }, 2);
  const before = JSON.stringify(s);
  assert.equal(stepVehicle(s, { throttle: 1, steer: 1, boost: false, action: true }, 0, FLAT), 0);
  assert.equal(JSON.stringify(s), before);
  // ...and a jump afterwards is still a finite arc
  stepVehicle(s, { throttle: 1, steer: 0, boost: false, action: true }, DT, FLAT);
  run(s, FLAT, { throttle: 1 }, 1);
  assert.ok(Number.isFinite(s.y) && Number.isFinite(s.vy));
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
  const canoe = run(createVehicleState('canoe', 0, 0, EAST, land), land, { throttle: 1 }, 16);
  assert.ok(canoe.x > 250, 'downriver and out onto the lake');
  const ashore = run(createVehicleState('canoe', 0, 0, 0, land), land, { throttle: 1, steerKey: true }, 3);
  assert.ok(Math.abs(ashore.z) < 8.5 && ashore.blocked === 'terrain', 'but not up the bank');
  const boat = run(createVehicleState('boat', 300, 0, -EAST, land), land, { throttle: 1 }, 8);
  assert.ok(boat.x > 199, 'the boat stays on the lake');
});

test('a canoe keeps going down a narrow, winding, cluttered creek', () => {
  // A creek only 5 wide, snaking east (centre line z = 14 sin(x / 40)), with a strong mask only
  // in mid-channel, a 6-long gravel bar across it at x = 150 and reeds thinning the mask
  const centre = (x) => 14 * Math.sin(x / 40);
  const creek = world(() => 30, {
    river: (x, z) => {
      if (x > 150 && x < 156) return 0;
      const off = Math.abs(z - centre(x));
      return off < 1 ? 0.8 : off < 2.5 ? 0.2 : 0;
    },
  });
  const s = createVehicleState('canoe', 0, 0, EAST, creek);
  let stuck = 0;
  let bent = 0;
  let worst = 0;
  for (let i = 0; i < 40 / DT && s.x < 400; i++) {
    stepVehicle(s, { throttle: 1, steer: 0, boost: false, action: false }, DT, creek);
    if (s.blocked) stuck++;
    bent += Math.abs(s.bent);
    if (!(s.x > 148 && s.x < 158)) worst = Math.max(worst, Math.abs(s.z - centre(s.x)));
  }
  assert.ok(s.x >= 400, `reached x=${s.x} (blocked on ${stuck} steps)`);
  assert.ok(worst < 6, `stayed in or beside the channel (strayed ${worst})`);
  assert.ok(bent > 1, 'it turned with the creek by itself');
  // Steering by key turns that help off: aimed at the bank, it stops
  const manual = createVehicleState('canoe', 0, 0, 0, creek);
  for (let i = 0; i < 4 / DT; i++) stepVehicle(manual, { throttle: 1, steer: 0, steerKey: true, boost: false, action: false }, DT, creek);
  assert.ok(Math.abs(manual.z) < 20 && manual.blocked === 'terrain', `held the key and ran aground at z=${manual.z}`);
  // A boat gets no such help
  assert.equal(summonSpot('boat', creek, 0, 0).ok, false);
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
