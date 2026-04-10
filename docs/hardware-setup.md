# Hardware Setup Guide

This guide covers wiring a Raspberry Pi for the AlpLion Smart Irrigation Controller.

> **Note:** Hardware selection is finalised in Phase 1.  The bill of materials below represents a recommended starting point based on cost and availability in Sri Lanka and Switzerland.

## Bill of Materials

| # | Component | Purpose | Est. cost (USD) |
| --- | --- | --- | --- |
| 1 | Raspberry Pi Zero 2 W | Main controller | ~$15 |
| 2 | Capacitive soil moisture sensor (e.g. STEMMA) | Soil moisture | ~$3 |
| 3 | DHT22 temperature/humidity sensor | Air temp & humidity | ~$4 |
| 4 | Rain detector module (digital output) | Rain detection | ~$2 |
| 5 | 5 V solenoid valve (NC, 1/2") | Water flow control | ~$8 |
| 6 | 1-channel 5 V relay module | Valve switching | ~$2 |
| 7 | ADS1115 16-bit ADC (I2C) | Analog sensor reading | ~$3 |
| 8 | 5 V / 2.5 A USB-C power supply | Power | ~$8 |
| 9 | 12 V DC submersible pump *(optional)* | Active water supply | ~$10 |
| 10 | Weatherproof enclosure | Field protection | ~$15 |

**Total (without pump): ~$60**

## GPIO Pin Assignments (BCM numbering)

| Signal | GPIO Pin | Notes |
| --- | --- | --- |
| Soil moisture sensor (digital) | GPIO 4 | Active LOW when wet |
| DHT22 data | GPIO 17 | 4.7 kΩ pull-up to 3.3 V |
| Rain sensor (digital) | GPIO 27 | Active LOW when raining |
| Valve relay control | GPIO 22 | HIGH = valve open |
| Pump relay control | GPIO 23 | HIGH = pump on |
| ADS1115 SDA | GPIO 2 (I2C SDA) | I2C bus |
| ADS1115 SCL | GPIO 3 (I2C SCL) | I2C bus |

## Wiring Diagrams

### Soil Moisture Sensor (Capacitive + ADS1115 ADC)

```
Raspberry Pi 3.3 V ──▶ VCC (sensor)
Raspberry Pi GND    ──▶ GND (sensor)
Sensor AOUT         ──▶ ADS1115 A0
ADS1115 VDD         ──▶ Raspberry Pi 3.3 V
ADS1115 GND         ──▶ Raspberry Pi GND
ADS1115 SDA         ──▶ Raspberry Pi GPIO 2 (SDA)
ADS1115 SCL         ──▶ Raspberry Pi GPIO 3 (SCL)
```

### DHT22 Temperature/Humidity Sensor

```
Raspberry Pi 3.3 V ──▶ VCC (pin 1)
                        Data (pin 2) ──▶ Raspberry Pi GPIO 17
                                     └── 4.7 kΩ ──▶ 3.3 V
Raspberry Pi GND    ──▶ GND (pin 4)
```

### Rain Detector Module

```
Raspberry Pi 5 V   ──▶ VCC
Raspberry Pi GND   ──▶ GND
Module D0 output   ──▶ Raspberry Pi GPIO 27
```

### Solenoid Valve via Relay

```
Raspberry Pi GPIO 22 ──▶ Relay IN
Raspberry Pi 5 V     ──▶ Relay VCC
Raspberry Pi GND     ──▶ Relay GND

12 V supply (+) ──▶ Relay COM
Relay NO        ──▶ Valve (+)
12 V supply (−) ──▶ Valve (−)
```

> ⚠️ **Safety:** Always use a relay or MOSFET to switch the solenoid valve.  Do not connect a solenoid valve directly to a GPIO pin.

## Raspberry Pi OS Setup

### Enable I2C

```bash
sudo raspi-config
# Interface Options → I2C → Enable
```

### Install system packages

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-smbus i2c-tools libgpiod2
```

### Verify ADS1115 I2C connection

```bash
i2cdetect -y 1
# Should show address 0x48 (ADS1115 default)
```

### Install the irrigation controller

```bash
git clone https://github.com/AlpLion-CH-LK/smart-irrigation-controller.git
cd smart-irrigation-controller
pip install -e ".[hardware]"
```

### Run as a system service (optional)

Create `/etc/systemd/system/irrigation.service`:

```ini
[Unit]
Description=AlpLion Smart Irrigation Controller
After=network.target

[Service]
ExecStart=/usr/local/bin/irrigation-controller run --region sri-lanka
WorkingDirectory=/home/pi/smart-irrigation-controller
Restart=always
User=pi

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable irrigation
sudo systemctl start irrigation
```

## Sensor Calibration

### Soil Moisture Sensor

The raw ADC output varies between sensors and soil types. Calibrate by:

1. Recording the ADC voltage when the sensor is in **completely dry** soil → set as `dry_voltage` in config.
2. Recording the ADC voltage when the sensor is in **fully saturated** soil → set as `wet_voltage` in config.

```yaml
# config.yaml
sensor:
  soil_moisture_pin: 4
  adc_channel: 0
```

In the Python API:

```python
from irrigation.sensors.soil import SoilMoistureSensor

sensor = SoilMoistureSensor(
    pin=4,
    adc_channel=0,
    dry_voltage=3.3,   # measured in dry soil
    wet_voltage=1.2,   # measured in saturated soil
)
```
