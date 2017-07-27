import math

"""Defines functions for low level PID controller."""
LENGTH_STIFFNESS = 1e4
ANGLE_MOTOR_RATIO = 16


def clamp(x, lower, upper):
    """Clip the output."""
    return min(max(x, lower), upper)


def pd_controller(err, derr, kp, kd):
    """Return PD equation."""
    return (kp * err) + (kd * derr)


def detect_stance(state):
    """Check if the robot's foot is on the ground."""
    l_comp = state.l_eq - state.l

    if l_comp < 0:
        return 'flight'

    elif l_comp > 0.01 and state.dl < 0:
        return 'stance'


def controller(state,
               leg_extension,
               target_velocity,
               horizontal_push):
    """Generate motor values from current state."""
    cstate = detect_stance(state)

    l_eq_target = 0.73 + leg_extension if state.dy > 0 else 0.7
    l_torque = pd_controller(
        l_eq_target - state.l_eq,
        0.0 - state.dl_eq,
        1e4,
        1e2
    )

    if(cstate == 'flight'):
        theta_eq_target = (state.dx * 0.2) - (target_velocity * 0.1) - state.phi
        dtheta_eq_target = 0.0
        theta_torque = pd_controller(
            theta_eq_target - state.theta_eq,
            dtheta_eq_target - state.dtheta_eq,
            1e4,
            1e2
        )

    else:
        theta_torque = pd_controller(state.phi, state.dphi, 1e3, 1e2) - horizontal_push
        l_force = LENGTH_STIFFNESS * (state.l_eq - state.l)
        friction_limit = 0.5 * max(l_force, 0) / state.l / ANGLE_MOTOR_RATIO
        theta_torque = clamp(theta_torque, -friction_limit, friction_limit)

    l_torque = 30 * math.tanh(l_torque/200)
    theta_torque = 30 * math.tanh(theta_torque/50)
    #return clamp(l_torque, -1e1, 1e1), clamp(theta_torque, -1e1, 1e1)
    return l_torque, theta_torque
