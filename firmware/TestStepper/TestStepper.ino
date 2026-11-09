#include <Arduino.h>

// Isolated DRV8825 test: no encoder, controller, or stepper library.
// Physical assumption: 200 steps/rev, MODE0/1/2 HIGH (1/32).
const uint8_t STEP_PIN = 9, DIR_PIN = 2, ENABLE_PIN = 5;
const uint16_t MOVE_PULSES = 533; // 29.98125 degrees at 6400 pulses/rev
const uint16_t RATES[] = {128, 128, 256, 256};
const uint32_t RUN_TIMEOUT_MS = 20000;

bool active = false, moving = false;
uint8_t stage = 0;
uint16_t pulses = 0;
int16_t net_pulses = 0;
uint32_t run_start_ms, phase_start_ms, last_step_us, move_start_us;
uint32_t max_late_us = 0;

void disableMotor()
{
    digitalWrite(ENABLE_PIN, HIGH);
    digitalWrite(STEP_PIN, LOW);
    active = false;
    moving = false;
}

void help()
{
    Serial.println(F("\nTestStepper v1 | Nano D9=STEP D2=DIR D5=EN(active LOW) | 115200 baud"));
    Serial.println(F("Assumes DRV8825, 200 steps/rev, MODE0/1/2 HIGH: 6400 pulses/rev."));
    Serial.println(F("RUN: hold 1s; +30/-30 deg at 128 Hz, then +30/-30 at 256 Hz; disable."));
    Serial.println(F("No encoder required. Counts are generated pulses, NOT measured movement."));
    Serial.println(F("Before r: center arm, clear travel, verify 1/32 and current limit; motor power ON."));
    Serial.println(F("r=run once  x=stop/disable  ?=help (idle only). No automatic repeat."));
}

void beginMove()
{
    const bool forward = (stage % 2 == 0);
    digitalWrite(DIR_PIN, forward ? HIGH : LOW);
    pulses = 0;
    max_late_us = 0;
    Serial.print(F("[MOVE "));
    Serial.print(stage + 1);
    Serial.print(F("/4] DIR="));
    Serial.print(forward ? F("HIGH +") : F("LOW -"));
    Serial.print(MOVE_PULSES);
    Serial.print(F(" pulses @ "));
    Serial.print(RATES[stage]);
    Serial.println(F(" Hz; watch smooth motion and direction."));
    // Finish printing before timing pulses. First pulse is one period later,
    // leaving ample DIR setup time. Slow pulses deliberately avoid the planner.
    Serial.flush();
    move_start_us = last_step_us = micros();
    moving = true;
}

void setup()
{
    digitalWrite(ENABLE_PIN, HIGH); // Set latch before making pin an output.
    pinMode(ENABLE_PIN, OUTPUT);
    pinMode(STEP_PIN, OUTPUT);
    pinMode(DIR_PIN, OUTPUT);
    disableMotor();
    Serial.begin(115200);
    help();
    Serial.println(F("[IDLE] EN=HIGH; motor disabled."));
}

void loop()
{
    if (Serial.available())
    {
        const char cmd = Serial.read();
        if (cmd == 'x' || cmd == 'X')
        {
            disableMotor();
            Serial.print(F("[STOP] EN=HIGH; generated net pulses="));
            Serial.println(net_pulses);
            Serial.println(F("Recenter with motor power OFF before another run."));
        }
        else if (!active && (cmd == 'r' || cmd == 'R'))
        {
            // Discard pasted/queued commands; never queue another run.
            while (Serial.available()) Serial.read();
            stage = 0;
            net_pulses = 0;
            active = true;
            moving = false;
            digitalWrite(ENABLE_PIN, LOW);
            phase_start_ms = run_start_ms = millis();
            Serial.println(F("[HOLD] EN=LOW for 1s; expect holding torque. x stops anytime."));
        }
        else if (!active && cmd == '?') help();
        // Ignore all other input, including line endings; no pulse-time logging.
    }
    if (!active) return;
    if ((uint32_t)(millis() - run_start_ms) >= RUN_TIMEOUT_MS)
    {
        disableMotor();
        Serial.println(F("[FAIL] Run exceeded 20s; disabled. Check timing/serial input."));
        return;
    }
    if (!moving)
    {
        if ((uint32_t)(millis() - phase_start_ms) >= 1000) beginMove();
        return;
    }

    const uint32_t period_us = 1000000UL / RATES[stage];
    const uint32_t now = micros();
    const uint32_t elapsed = now - last_step_us;
    if (elapsed < period_us) return;
    const uint32_t late = elapsed - period_us;
    if (late > max_late_us) max_late_us = late;
    last_step_us = now; // Never emit a burst to catch up after a delay.
    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(5);
    digitalWrite(STEP_PIN, LOW);
    ++pulses;
    net_pulses += (stage % 2 == 0) ? 1 : -1;
    if (pulses < MOVE_PULSES) return;

    const uint32_t duration_us = micros() - move_start_us;
    moving = false;
    ++stage;
    if (stage == 4) disableMotor(); // Disable before summary serial writes.
    Serial.print(F("[PULSES] sent="));
    Serial.print(pulses);
    Serial.print(F(" elapsed_ms="));
    Serial.print(duration_us / 1000);
    Serial.print(F(" avg_Hz="));
    Serial.print((float)pulses * 1000000.0f / duration_us, 1);
    Serial.print(F(" max_late_us="));
    Serial.print(max_late_us);
    Serial.print(F(" net="));
    Serial.println(net_pulses);
    if (stage == 4)
    {
        Serial.println(F("[DONE] 2132 pulses generated; net=0; EN=HIGH (disabled)."));
        Serial.println(F("Physical result UNVERIFIED: both directions smooth, ~30 deg, returned to start?"));
        Serial.println(F("No motion: check VMOT/GND, EN, RESET/SLEEP, coils and current limit."));
        Serial.println(F("Buzz/stall: coil pairing, current, load. Wrong angle: MODE pins/shaft slip."));
        Serial.println(F("Share this log + observed motion/angle/noise/holding torque."));
    }
    else phase_start_ms = millis();
}
