import numpy as np


def estimate_power(torque, velocity, positive_only=False):
    # Specs taken from RoboDrive datasheet for ILM 115x50

    # in Newton-meters
    max_motor_torques = np.array([4.66, 4.66, 12.7, 12.7, 0.99,
                                  4.66, 4.66, 12.7, 12.7, 0.99])

    # in Watts
    power_loss_at_max_torque = np.array([19.3, 19.3, 20.9, 20.9, 5.4,
                                         19.3, 19.3, 20.9, 20.9, 5.4])

    gear_ratios = np.array([25, 25, 16, 16, 50,
                            25, 25, 16, 16, 50])

    # calculate power loss constants
    power_loss_constants = power_loss_at_max_torque / np.square(max_motor_torques)

    # get output torques and velocities
    output_torques = np.array(torque)
    output_velocity = np.array(velocity)

    # calculate input torques
    input_torques = output_torques / gear_ratios

    # get power loss of each motor
    power_losses = power_loss_constants * np.square(input_torques)

    # calculate positive motor power only for each motor
    motor_powers = output_torques * output_velocity

    # replace negative powers w/ zero
    if positive_only:
        motor_powers[motor_powers < 0] = 0

    # estimate power
    return np.sum(motor_powers) + np.sum(power_losses), {'input_torques': input_torques, 'motor_powers': motor_powers}