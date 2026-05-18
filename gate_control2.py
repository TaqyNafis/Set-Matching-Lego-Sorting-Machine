from gpiozero import Servo
from gpiozero.tones import Tone
from signal import pause
from time import sleep
import sys
import warnings
warnings.filterwarnings("ignore")

servo = Servo(18, min_pulse_width=0.0007, max_pulse_width=0.0023, initial_value=None)

def move(angle):
    angle = max(0, min(180, angle))
    value = (angle - 90) / 90
    print(f"Setting servo value to {value:.3f} for angle {angle}")
    servo.value = value
    sleep(0.8)
    servo.detach()
    print("Done")

if len(sys.argv) != 2:
    print("Usage: python3 gate_control2.py <angle>")
    exit(1)

try:
    angle = float(sys.argv[1])
except ValueError:
    print("Angle must be a number")
    exit(1)

print(f"Moving to {angle}")
move(angle)

servo.close()
