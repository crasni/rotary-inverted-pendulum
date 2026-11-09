# Hardware test commands

Run these from the repository root, in order. Keep **12 V OFF while uploading,
wiring, or centering the arm**. USB stays connected. Close each serial monitor
with **Ctrl-C** before the next upload or Python command. Stop at a failed check.

## 0. Terminal setup

```bash
source .venv/bin/activate
pendulum-lab ports
export RIG_PORT=/dev/ttyUSB0  # replace with your listed port
```

If setting up a new environment, first run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,firmware]'
```

## 1. TestHeartbeat — USB and board

```bash
pio run -d firmware/TestHeartbeat -t upload --upload-port "$RIG_PORT"
pio device monitor --port "$RIG_PORT" --baud 115200
```

**Expected:** onboard LED double-blinks. This checks the board, not the motor.
Press Ctrl-C to exit the monitor.

## 2. TestEncoder — sensor only

```bash
pio run -d firmware/TestEncoder -t upload --upload-port "$RIG_PORT"
pio device monitor --port "$RIG_PORT" --baud 115200
```

**Expected:** I2C/magnet detected; angle changes when you turn the pendulum by
hand, with 12 V off. If it reports no I2C response or no magnet, fix that before
using LowLevelServer. Press Ctrl-C to exit.

## 3. TestStepper — isolated driver and motor

No encoder is needed for this sketch. Verify the
[wiring and current limit](docs/hardware.md#wiring), especially **MODE0/1/2 HIGH
for 1/32 microstepping**. Wrong mode settings can cause much larger travel.
Center the arm with 12 V off and clear its travel.

```bash
pio run -d firmware/TestStepper -t upload --upload-port "$RIG_PORT"
pio device monitor --port "$RIG_PORT" --baud 115200
```

Switch **12 V ON**, then type **`r`** to run once (Enter is harmless).
Type **`x`** to stop/disable; use the physical power switch if needed.

**Expected:** one-second hold, approximately +30° and back at 128 Hz, then
+30° and back at 256 Hz. The driver automatically disables at the end (~17 s).
The log prints counts/timing per move, then `[DONE]`. That confirms generated
pulses only; watch for smooth motion, reversal, and return to the starting mark.

Copy the log and record:

```text
Initial holding torque: yes / no / unsure
128 Hz: smooth / buzz / no motion; approximate angle: ___
256 Hz: smooth / buzz / no motion; approximate angle: ___
Reversed and returned to start: yes / no
```

No motion: check motor power, GND, ENABLE, RESET/SLEEP, coils and current limit.
Buzzing: check coil pairing/current/mechanical binding. Wrong angle: check
MODE pins and shaft coupling. Never unplug the motor while powered.

Switch **12 V OFF**, then Ctrl-C. Recenter before another test.
More detail: [TestStepper guide](firmware/TestStepper/README.md).

## 4. Restore LowLevelServer — normal firmware preflight

With 12 V off, arm centered and pendulum hanging still:

```bash
pio run -d firmware/LowLevelServer -t upload --upload-port "$RIG_PORT"
pendulum-lab preflight --port "$RIG_PORT"
```

**Expected:** preflight passes with protocol 3, 32 microsteps, 500000 baud,
and valid feedback. This firmware needs the working encoder. It uses a binary
protocol: do not use the text serial monitor to operate it.

## 5. Final motor check through normal firmware

Start with **12 V OFF**, arm centered, pendulum hanging still:

```bash
pendulum-lab motor-check --port "$RIG_PORT"
```

Follow the prompt: switch 12 V on and type **`MOVE`**. Expect one finite
+533-microstep move (~30°), then automatic disable. Verify actual motion agrees.
Switch **12 V OFF** before touching or recentering the arm.

If both motor tests pass, the basic motor commissioning is complete.

## 6. Optional: test a controller after commissioning

Use your identified rig parameters and a controller already verified in
simulation. Example for LQR (replace the parameter path with your own):

```bash
pendulum-lab sim --algorithm lqr --params configs/my_rig.json --start hanging --duration 3 --output runs/lqr_sim_check
```

After successful simulation and the commissioning checks, start with 12 V off,
arm centered and pendulum hanging still:

```bash
pendulum-lab hardware --algorithm lqr --params configs/my_rig.json --port "$RIG_PORT" --duration 3 --output runs/lqr_hw_check
```

At the prompt, switch power on and type **`RUN`**. Keep the swing area clear and
power switch accessible. Turn power off afterward. Inspect the output directory's
`summary.json`; a planned duration stop alone does not mean successful balancing.
Use a fresh output directory for each run. See [deployment](docs/deployment.md)
for the full controller workflow.
