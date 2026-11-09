#include <AS5600.h>
#include <Wire.h>

// =============================================================================
// CONFIGURATION (lab wiring; motor output remains disabled)
// =============================================================================
const long SERIAL_BAUD_RATE = 115200;  // Same monitor speed as TestHeartbeat
const long I2C_CLOCK_HZ = 100000;      // Conservative speed for wiring diagnostics
const uint32_t I2C_TIMEOUT_US = 25000;

// Motor pins (to disable motor and prevent heating)
const int DIR_PIN = 2;
const int STEP_PIN = 9;     // Timer1 OC1A (rig wired here so FastAccelStepper-based sketches work too)
const int ENABLE_PIN = 5;

// =============================================================================
// STATE
// =============================================================================
AS5600 as5600;

// Plotting variables
uint32_t last_plot_ms = 0;
const uint32_t PLOT_INTERVAL_MS = 50;

void setup()
{
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, HIGH);  // LED on during setup

    // Disable motor driver to prevent heating
    pinMode(ENABLE_PIN, OUTPUT);
    digitalWrite(ENABLE_PIN, HIGH);  // HIGH = disabled (active-low enable)
    pinMode(DIR_PIN, OUTPUT);
    pinMode(STEP_PIN, OUTPUT);

    Serial.begin(SERIAL_BAUD_RATE);
    Wire.begin();
    Wire.setWireTimeout(I2C_TIMEOUT_US, true);
    Wire.setClock(I2C_CLOCK_HZ);
    as5600.begin();

    // A missing magnet and a failed I2C read must not look like the same fault.
    while (true)
    {
        Wire.clearWireTimeoutFlag();
        as5600.lastError(); // Clear any error from the preceding attempt.
        if (!as5600.isConnected())
        {
            Serial.print(F("[AS5600] No I2C response at 0x36; check sensor power, GND, SDA/A4, SCL/A5. timeout="));
            Serial.println(Wire.getWireTimeoutFlag());
            delay(1000);
            continue;
        }

        const uint8_t status = as5600.readStatus();
        const int error = as5600.lastError();
        if (error || Wire.getWireTimeoutFlag())
        {
            Serial.print(F("[AS5600] I2C status read failed: error="));
            Serial.print(error);
            Serial.print(F(" timeout="));
            Serial.println(Wire.getWireTimeoutFlag());
        }
        else if (status & (1U << 5)) // MD: magnet detected, STATUS bit 5.
        {
            Serial.println(F("[AS5600] I2C OK; magnet detected."));
            break;
        }
        else
        {
            Serial.print(F("[AS5600] I2C OK; waiting for magnet. Center the correct magnet over the chip. status=0x"));
            Serial.println(status, HEX);
        }
        delay(1000);
    }

    // Print the current magnitude of the magnet
    Serial.print("[AS5600] Current magnitude: ");
    Serial.println(as5600.readMagnitude());

    // Print the magnet strength
    if (as5600.magnetTooWeak())
    {
        Serial.println("[AS5600] Magnet strength is too weak. ---");
    }
    else if (as5600.magnetTooStrong())
    {
        Serial.println("[AS5600] Magnet strength is too strong. +++");
    }
    else
    {
        Serial.println("[AS5600] Magnet strength is just right!");
    }

    digitalWrite(LED_BUILTIN, LOW);  // LED off when ready
    Serial.println("[AS5600] Encoder test ready - rotate the pendulum");
    Serial.println("[Motor] Driver disabled (not heating)");
}

void loop()
{
    // Get the pendulum position
    float pendulum_actual_deg = convertRawAngleToDegrees();

    // Print at 20 Hz, independently of the sensor read rate.
    const uint32_t now_ms = millis();
    if ((uint32_t)(now_ms - last_plot_ms) >= PLOT_INTERVAL_MS)
    {
        last_plot_ms = now_ms;
        // Flash LED briefly to indicate activity
        digitalWrite(LED_BUILTIN, HIGH);

        // Format for Arduino Serial Plotter: "label:value"
        Serial.print("pendulum_deg:");
        Serial.println(pendulum_actual_deg);

        digitalWrite(LED_BUILTIN, LOW);
    }
    delay(5);
}

/*
 * Convert the raw angle from the AS5600 magnetic encoder to degrees.
 * Handles multi-revolution tracking with wraparound detection.
 */
float convertRawAngleToDegrees()
{
    // AS5600 constants
    const long AS5600_RESOLUTION = 4096;
    const long WRAPAROUND_THRESHOLD = AS5600_RESOLUTION / 2;  // 2048
    const float DEG_PER_SEGMENT = 360.0f / AS5600_RESOLUTION; // 0.087890625

    static long raw_prev = 0;
    static bool first_reading = true;
    static float position = 0.0f;

    // Never accumulate failed reads as real movement. Stop until reset so a
    // recovered connection cannot hide an unknown amount of pendulum motion.
    const bool magnet = as5600.magnetDetected();
    const int status_error = as5600.lastError();
    long raw = as5600.rawAngle();
    const int angle_error = as5600.lastError();
    if (!magnet || status_error || angle_error || Wire.getWireTimeoutFlag())
    {
        Serial.print("[AS5600] Read failed: magnet=");
        Serial.print(magnet);
        Serial.print(" status_error=");
        Serial.print(status_error);
        Serial.print(" angle_error=");
        Serial.print(angle_error);
        Serial.print(" timeout=");
        Serial.println(Wire.getWireTimeoutFlag());
        Serial.println("[AS5600] Check power, GND, SDA/A4, SCL/A5 and magnet; reset after fixing.");
        digitalWrite(LED_BUILTIN, HIGH);
        while (true) delay(1000); // Motor remains disabled.
    }

    if (first_reading)
    {
        raw_prev = raw;
        first_reading = false;
    }

    long delta = raw - raw_prev;

    // Handle wraparound (movement > 180 deg between samples indicates wraparound)
    if (delta > WRAPAROUND_THRESHOLD)  delta -= AS5600_RESOLUTION;
    if (delta < -WRAPAROUND_THRESHOLD) delta += AS5600_RESOLUTION;

    position += (float)delta * DEG_PER_SEGMENT;

    // Save the current raw angle for the next iteration
    raw_prev = raw;

    return position;
}
