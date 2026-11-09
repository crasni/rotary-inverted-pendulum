# Isolated stepper-driver test

Tests the Nano → DRV8825 → motor path without AS5600, control algorithms,
or FastAccelStepper. Uses the existing D9 STEP, D2 DIR, D5 active-low ENABLE
wiring. Flashing replaces the current sketch; restore LowLevelServer afterward.

From the repository root, with motor power off:

```bash
.venv/bin/pio run -d firmware/TestStepper
.venv/bin/pio run -d firmware/TestStepper -t upload --upload-port /dev/ttyUSB0
.venv/bin/pio device monitor -p /dev/ttyUSB0 -b 115200
```

Replace the port as needed. Arduino IDE also works: open `TestStepper.ino`,
select the Nano board/port, upload, and open Serial Monitor at **115200**.
No external libraries are needed.

1. Follow the existing [wiring/current-limit checks](../../docs/hardware.md#wiring).
   Verify MODE0/1/2 are HIGH (1/32), RESET/SLEEP HIGH, common ground, and coil pairs.
   Incorrect microstepping can make travel much larger than 30 degrees.
2. Center the arm with motor power off, leave travel clear, then switch motor
   power on. Opening the monitor does not start motion.
3. Send `r` once. Expect a one-second hold, then approximately +30°, back,
   +30°, back, with one-second pauses. The second pair is twice as fast.
   Positive means DIR HIGH; the physical clockwise direction depends on wiring.
4. Send `x` to abort; use the motor power switch if communication is lost.
   The finite sequence disables the driver after about 17 seconds, with a
   20-second run timeout. After interruption, recenter with power off before retrying.

The console prints one start and one timing line per move, then a summary.
Expected durations are about **4164 ms at 128 Hz** and **2082 ms at 256 Hz**.
`max_late_us` is the largest software scheduling delay beyond the pulse period;
delays comparable to the period indicate uneven generation. There is no stream
of per-pulse messages. The sketch uses slow, constant-rate pulses without ramps.

| Observation | What it suggests checking |
| --- | --- |
| No hold and no motion | VMOT, common ground, ENABLE at driver, RESET/SLEEP, current limit, coil continuity |
| Holds but buzzes/stalls | Coil pairing, current limit, mechanical binding/load; inspect STEP at driver |
| Only one physical direction | DIR connection/level at the driver |
| Wrong travel angle | MODE settings, motor steps/revolution, shaft coupling slip |
| Does not return to start | Missed steps, binding, loose coupling |
| This test works, normal firmware fails | Encoder safety faults, stepper planner, serial protocol/configuration |

`[DONE]` confirms only that software generated the sequence. The sketch cannot
read driver power, current, temperature, fault output (not wired), or shaft
position. EN/DIR labels describe commanded MCU levels, not measured driver
terminals. It does not certify the driver electrically or validate high-speed
operation. Never disconnect motor wiring while powered.

To share results, copy the console log and add:

```text
Holding torque during initial hold: yes / no / unsure
128 Hz: smooth / buzz / no motion; outward angle roughly ___ degrees
256 Hz: smooth / buzz / no motion; outward angle roughly ___ degrees
Reversed and returned to start: yes / no
Other observations: ___
```
