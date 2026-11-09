#include <AS5600.h>
#include <Wire.h>

// Read-only AS5600 field-strength monitor (diagnostics; motor stays disabled).
//
// Unlike TestEncoder it never stops on a failed read: it counts failures and
// keeps going, so you can move the pendulum and arm and watch how the magnet
// magnitude, the chip's automatic gain (AGC) and the read errors change.
//
// Every 50 ms it prints one line:
//   angle_deg:<0-360> magnitude:<0-4095> agc:<0-255> md:<0/1> ml:<0/1> mh:<0/1> errors:<total>
//   md = magnet detected, ml = too weak, mh = too strong (STATUS bits 5, 4, 3).
// AGC near 0 means the field is strong (magnet close); near 255 means weak
// (far). It should stay well inside both ends, and steady as the pendulum
// turns. Every 2 s a summary gives the min/max magnitude and AGC and the
// error count since the previous summary.

const long SERIAL_BAUD_RATE = 115200;
const long I2C_CLOCK_HZ = 400000;      // same bus speed as LowLevelServer
const uint32_t I2C_TIMEOUT_US = 1500;  // same timeout as LowLevelServer

const int DIR_PIN = 2;
const int STEP_PIN = 9;
const int ENABLE_PIN = 5;

AS5600 as5600;

uint32_t last_print_ms = 0;
uint32_t last_summary_ms = 0;
uint32_t total_errors = 0;
uint32_t window_errors = 0;
uint16_t mag_min = 4095, mag_max = 0;
uint8_t agc_min = 255, agc_max = 0;

void setup()
{
    pinMode(ENABLE_PIN, OUTPUT);
    digitalWrite(ENABLE_PIN, HIGH);  // driver disabled (active-low enable)
    pinMode(DIR_PIN, OUTPUT);
    pinMode(STEP_PIN, OUTPUT);

    Serial.begin(SERIAL_BAUD_RATE);
    Wire.begin();
    Wire.setWireTimeout(I2C_TIMEOUT_US, true);
    Wire.setClock(I2C_CLOCK_HZ);
    as5600.begin();
    Serial.println(F("[TestMagnitude] read-only; motor disabled. Move the pendulum and arm."));
}

void loop()
{
    Wire.clearWireTimeoutFlag();
    as5600.lastError();  // clear the previous error

    const uint8_t status = as5600.readStatus();
    const int e1 = as5600.lastError();
    const uint16_t magnitude = as5600.readMagnitude();
    const int e2 = as5600.lastError();
    const uint8_t agc = as5600.readAGC();
    const int e3 = as5600.lastError();
    const uint16_t raw = as5600.rawAngle();
    const int e4 = as5600.lastError();
    const bool timeout = Wire.getWireTimeoutFlag();

    const bool failed = e1 || e2 || e3 || e4 || timeout;
    const bool md = status & (1U << 5);
    const bool ml = status & (1U << 4);
    const bool mh = status & (1U << 3);

    if (failed)
    {
        total_errors++;
        window_errors++;
        Serial.print(F("[READ ERROR] status_err="));
        Serial.print(e1);
        Serial.print(F(" magnitude_err="));
        Serial.print(e2);
        Serial.print(F(" agc_err="));
        Serial.print(e3);
        Serial.print(F(" angle_err="));
        Serial.print(e4);
        Serial.print(F(" timeout="));
        Serial.println(timeout);
    }
    else
    {
        if (magnitude < mag_min) mag_min = magnitude;
        if (magnitude > mag_max) mag_max = magnitude;
        if (agc < agc_min) agc_min = agc;
        if (agc > agc_max) agc_max = agc;
        if (!md)
        {
            window_errors++;
            total_errors++;
            Serial.println(F("[MAGNET LOST] md=0"));
        }
    }

    const uint32_t now_ms = millis();
    if (!failed && (uint32_t)(now_ms - last_print_ms) >= 50)
    {
        last_print_ms = now_ms;
        Serial.print(F("angle_deg:"));
        Serial.print(raw * (360.0f / 4096.0f), 1);
        Serial.print(F(" magnitude:"));
        Serial.print(magnitude);
        Serial.print(F(" agc:"));
        Serial.print(agc);
        Serial.print(F(" md:"));
        Serial.print(md);
        Serial.print(F(" ml:"));
        Serial.print(ml);
        Serial.print(F(" mh:"));
        Serial.print(mh);
        Serial.print(F(" errors:"));
        Serial.println(total_errors);
    }
    if ((uint32_t)(now_ms - last_summary_ms) >= 2000)
    {
        last_summary_ms = now_ms;
        Serial.print(F("[SUMMARY 2s] magnitude "));
        Serial.print(mag_min);
        Serial.print(F("-"));
        Serial.print(mag_max);
        Serial.print(F(" agc "));
        Serial.print(agc_min);
        Serial.print(F("-"));
        Serial.print(agc_max);
        Serial.print(F(" errors "));
        Serial.print(window_errors);
        Serial.print(F(" (total "));
        Serial.print(total_errors);
        Serial.println(F(")"));
        mag_min = 4095; mag_max = 0; agc_min = 255; agc_max = 0; window_errors = 0;
    }
    delay(2);  // about the main firmware's 500 Hz sampling
}
