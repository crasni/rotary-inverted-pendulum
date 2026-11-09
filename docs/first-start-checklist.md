# First startup: wiring to identification

Use this checklist in order. The team performs the inspections itself; a TA
sign-off is not required. This consolidates `hardware.md`, `deployment.md`, and
the start of `system-identification.md`. It does not authorize a controller trial.

Have a multimeter, a way to measure 30 degrees, a physical center mark, and an
accessible 12 V switch ready. Assign one person to the terminal and one to observe
motion and operate the switch. Record photos, readings, and pass/fail notes.

## 1. Inspect wiring and mechanics — USB and 12 V disconnected

- [ ] Verify the actual board is an ATmega328P Nano, driver is DRV8825, motor is
  200 full steps/revolution, and encoder is AS5600. If different, resolve the
  configuration before using the supplied commands.
- [ ] Identify the two motor coil pairs with a multimeter; connect one complete
  pair to A1/A2 and the other to B1/B2. Do not rely on wire colors.
- [ ] Check D9 → STEP, D2 → DIR, D5 → ENABLE, A4 → SDA, A5 → SCL.
- [ ] Verify common grounds and sensor supply voltage against the actual breakout.
- [ ] Verify switched 12 V goes to driver VMOT and Nano VIN, never Nano 5 V or
  sensor supply pins. Check carrier orientation and pin labels on the actual board.
- [ ] MODE0/1/2 are all high for 1/32 microsteps; RESET and SLEEP are logic high.
- [ ] Check solder joints, polarity, exposed conductors, and accidental shorts.
- [ ] Verify supply decoupling at VMOT/GND. The existing hardware guide specifies
  100 nF ceramic plus 22 µF electrolytic, but the
  [Pololu DRV8825 carrier guidance](https://www.pololu.com/product/2133)
  recommends at least 47 µF electrolytic near the board. Resolve this for your
  actual carrier before energizing; use correct polarity and a suitable voltage rating.
- [ ] Secure motor, arm, bearing, pendulum, magnet, sensor, and base. Confirm free
  pendulum swing without rubbing and wire clearance throughout ±100 degrees of arm travel.
- [ ] Mark physical arm center and a 30-degree reference. Weigh the complete pendulum.

## 2. Set driver current limit — motor disconnected

This is the one setup step that powers the driver before the motor is connected.

1. With all power disconnected, disconnect the motor from the driver.
2. Follow the actual carrier's current-setting procedure. Power the driver and
   measure Vref relative to driver GND without shorting adjacent contacts.
3. The repository target is **0.45 V**, approximately **0.9 A current limit for
   0.1-ohm sense resistors**. Confirm the actual carrier's resistor value and the
   motor rating; this conversion is not universal across clone boards.
4. Switch 12 V off, disconnect USB if attached, allow stored supply charge to
   discharge, then reconnect the motor and recheck wiring.
5. Leave 12 V off for steps 3–5 below.

Never connect or disconnect the motor while its driver is powered.

## 3. Check USB and heartbeat — 12 V OFF

From the repository root, with dependencies already installed:

```bash
source .venv/bin/activate
pendulum-lab ports
```

Connect USB, physically center the arm, and let the pendulum hang still.
Replace `/dev/ttyUSB0` in every command with the detected port. If no port is
visible, resolve USB access before proceeding; Windows/WSL may require USB
attachment to WSL or running the tooling on Windows.

```bash
pio run -d firmware/TestHeartbeat -t upload --upload-port /dev/ttyUSB0
pio device monitor --port /dev/ttyUSB0 --baud 115200
```

- [ ] Upload succeeds, LED double-blinks, and heartbeat messages appear.
- [ ] Press Ctrl-C to close the monitor before the next upload.

The heartbeat only checks the board/USB/upload path. It does not test the encoder
or motor. Keep the motor supply off regardless of which sketch is installed.

## 4. Check encoder — 12 V OFF

```bash
pio run -d firmware/TestEncoder -t upload --upload-port /dev/ttyUSB0
pio device monitor --port /dev/ttyUSB0 --baud 115200
```

- [ ] It reaches `Encoder test ready`, rather than staying at `Waiting for magnet`.
- [ ] Magnet strength is reported as `just right`; resolve weak/strong readings.
- [ ] Gently move the pendulum by hand while keeping the arm fixed. The printed
  `pendulum_deg` changes continuously and follows the motion.
- [ ] Return to the same hanging pose; readings return approximately to their
  previous hanging value and are stable at rest.
- [ ] Close the monitor with Ctrl-C, center the arm, and let the pendulum hang still.

The diagnostic sketch starts its angle at the first sample, so its zero is not
independent proof of hanging calibration. Start it with the pendulum hanging;
the identification workflow will record its own hanging tare later.

## 5. Install the operating firmware and preflight — 12 V OFF

```bash
pio run -d firmware/LowLevelServer -t upload --upload-port /dev/ttyUSB0
pendulum-lab preflight --port /dev/ttyUSB0
```

- [ ] The command prints `Preflight passed`.
- [ ] The arm is physically centered and pendulum is still and hanging.

This checks protocol 3, 32-microstep/500000-baud profile, fault status, fresh and
plausible feedback, and startup state. It cannot verify wiring, physical arm
centering, or actual motor movement. Opening serial may reset the board and
establish a new origin, so physically prepare the pose before every command.

The low-level server uses a binary protocol; use `pendulum-lab`, not a text
serial monitor, to inspect it. Only one program should own the port at a time.
For failed uploads, check port, cable, permissions, port ownership, and Nano
bootloader selection. Do not change controller gains to fix uploads.

## 6. One finite motor check — power ON only at the prompt

1. Start with 12 V off. Remove supports; center the arm and let the pendulum hang
   freely. Clear the entire swept area. Start recording video.
2. Run:

   ```bash
   pendulum-lab motor-check --port /dev/ttyUSB0
   ```

3. After preflight, follow the prompt: switch 12 V on, keep everyone clear, then
   type `MOVE`. The requested move is +533 microsteps, approximately +30 degrees.
4. Observe the actual motion against the marked angle. The command finishes by
   disabling the motor; software completion alone is not a physical pass.
5. Switch 12 V off before touching or manually returning the arm to center.

Pass only if the arm physically moves approximately 30 degrees in the intended
positive direction, without apparent skipping, slipping, or buzzing without
motion, and the software reports no failure. The firmware count can advance even
if the motor does not. Investigate any discrepancy with power off before retrying.
For unexpected motion, use the physical power switch. Inspect the cause of a
latched fault before resetting the board.

## 7. Begin motor-off system identification

With 12 V off, USB connected, arm centered and mechanically secured, and the
pendulum free to swing:

```bash
pendulum-lab sysid --port /dev/ttyUSB0 --skip-motor \
  --out-dir runs/sysid_train_01 --out-json configs/my_rig.json
```

Follow the hanging tare and three free-swing prompts, recording the physical
process on video. Use a new output directory and JSON filename if those exist.
Then follow `system-identification.md` for fit inspection, measured-mass entry,
independent free-swing validation, and subsequent powered response/coupling tests.

A passed first-start check is not evidence that a controller is ready. Complete
identification and validation before choosing and testing a hardware controller.
