# T200 thrust calibration

`T200_PWM_N.csv` is copied from
`bluerov_core/bridge/docs/T200_PWM_N.csv`. It contains measured Blue Robotics
T200 force in newtons versus PWM at 14, 16, and 18 volts. Keep both copies
byte-identical.

SHA-256: `9fe47f30c65a80180cef158ca096eee7fe227e012a56af102591fc748083df0d`

## ArduSub `VECTORED_6DOF` to MVM mapping

The MVM configuration numbers the left thruster before the right thruster at
each fore/aft location, while ArduSub `VECTORED_6DOF` numbers the corresponding
right thruster first. Each physical left/right pair is therefore swapped when
servo PWM is passed into MVM. The direction multipliers then convert the motor
orientation into the MVM wrench convention:

| MVM thruster | ArduSub servo | Direction |
| --- | ---: | ---: |
| T1 front-left horizontal | 2 | -1 |
| T2 front-right horizontal | 1 | -1 |
| T3 rear-left horizontal | 4 | +1 |
| T4 rear-right horizontal | 3 | +1 |
| T5 front-left vertical | 6 | -1 |
| T6 front-right vertical | 5 | -1 |
| T7 rear-left vertical | 8 | -1 |
| T8 rear-right vertical | 7 | -1 |

This table follows the physical left/right identification of the installed
vehicle. `config/BlueROV.conf` uses FLU coordinates and matching physical
left/right extrinsics. Its horizontal thrusters are located at `z=0` relative
to the model torque origin so ArduSub motor detection sees the yaw-only angular
factor expected by `VECTORED_6DOF`.
