"""LeRobot teleoperator for two SpaceMouse devices and RB-Y1."""

import sys
import time
from contextlib import ExitStack
from functools import wraps
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.teleoperators.teleoperator import Teleoperator

from .action_mapper import Rby1ActionMapper
from .config_spacemouse4rby1 import SpaceMouse4Rby1Config
from .devices import AXES, DualSpaceMouse, SingleLeftSpaceMouse
from .pose_integrator import DualPoseIntegrator
from .rby1_state import Rby1StateReader


class SpaceMouse4Rby1(Teleoperator):
    config_class = SpaceMouse4Rby1Config
    name = "spacemouse4rby1"

    def __init__(self, config):
        super().__init__(config)

        self.config = config

        if config.single_device:
            self.devices = SingleLeftSpaceMouse(config.left_device)
        else:
            self.devices = DualSpaceMouse(
                config.left_device,
                config.right_device,
            )

        self.state_reader = None
        self.integrator = DualPoseIntegrator(config)
        self.mapper = Rby1ActionMapper(config.use_gripper)

        self._stack = None
        self._follower = None
        self._initialized = False
        self._wait_neutral = True
        self._last_update = None
        self._was_active = {
            "left": False,
            "right": False,
        }
        self._rotation_only = {"left": False, "right": False}
        self._mode_button_pressed = {"left": False, "right": False}
        self._mode_wait_neutral = {"left": False, "right": False}

    @property
    def action_features(self):
        return self.mapper.action_features

    @property
    def feedback_features(self):
        return {}

    @property
    def is_connected(self):
        return self._stack is not None

    @property
    def is_calibrated(self):
        return True

    def connect(self, calibrate=True):
        if self.is_connected:
            raise RuntimeError("Teleoperator is already connected.")

        stack = ExitStack()

        try:
            stack.enter_context(self.devices)
        except BaseException:
            stack.close()
            raise

        self._stack = stack

        # robot.connect()에서 ready 동작이 실행될 수 있으므로
        # 여기서는 아직 목표 pose를 초기화하지 않는다.
        self._initialized = False
        self._wait_neutral = True
        self._last_update = None

        

    def bind_robot(self, robot):
        """기존 RBY1 robot의 connect()가 끝난 뒤 호출한다."""
        if not self.is_connected or not robot.is_connected:
            raise RuntimeError(
                "Connect both teleoperator and robot first."
            )

        if robot.name != "rby1":
            raise ValueError(
                "This teleoperator requires the RBY1 robot plugin."
            )

        if set(robot.action_features) != set(self.action_features):
            raise ValueError(
                "Use robot.action_mode=ee with both arms enabled, "
                "torso/mobile base disabled, and matching use_gripper."
            )

        if self._follower is robot and self.state_reader is not None:
            self.reset_targets()
            return
        if self._follower is not None:
            raise RuntimeError("Teleoperator is already bound to a robot.")

        # 별도 상태 조회 연결을 만들되 주소는 robot 설정에서 가져온다.
        reader = Rby1StateReader(robot._config.address, self.config.robot_model)
        reader.connect()
        try:
            self.state_reader = reader
            self._follower = robot
            self.reset_targets()
        except BaseException:
            reader.disconnect()
            self.state_reader = None
            self._follower = None
            raise
        self._stack.callback(reader.disconnect)

    def reset_targets(self):
        """현재 실제 pose와 그리퍼 위치를 목표로 다시 설정한다."""
        if not self.is_connected:
            raise RuntimeError("Teleoperator is not connected.")

        if self.state_reader is None:
            raise RuntimeError("Call bind_robot(robot) before reset_targets().")
        self._initialized = False
        grippers = None

        if self.config.use_gripper:
            if self._follower is None:
                raise RuntimeError(
                    "Call bind_robot(robot) before requesting "
                    "gripper actions."
                )

            obs = self._follower.get_observation()

            grippers = {
                side: obs[f"{side}_gripper_0.pos"]
                for side in ("left", "right")
            }

        self.integrator.reset(
            self.state_reader.read(),
            grippers,
        )

        self._initialized = True
        self._wait_neutral = True
        self._last_update = None
        self._was_active = {
            "left": False,
            "right": False,
        }



    def _is_neutral(self, inputs):
        for raw in inputs.values():
            axes = np.array(
                [raw[key] for key in AXES],
                dtype=float,
            )

            if not np.isfinite(axes).all():
                raise ValueError("Invalid SpaceMouse axes.")

            if (
                np.any(np.abs(axes) > self.config.deadzone)
                or any(raw["buttons"])
            ):
                return False

        return True

    def get_action(self):
        if not self.is_connected:
            raise RuntimeError("Teleoperator is not connected.")

        if not self._initialized:
            self.reset_targets()

        # Linux에서 직접 연결된 장치가 제거되었는지 확인한다.
        for path in self.devices.paths.values():
            if not Path(path).exists():
                raise ConnectionError(
                    f"SpaceMouse path disappeared: {path}"
                )

        inputs = self.devices.read()
        now = time.perf_counter()

        # 루프가 오래 멈췄다면 이전 목표를 버리고 다시 초기화한다.
        if (
            self._last_update is not None
            and now - self._last_update > 0.5
        ):
            self.reset_targets()

        if self._wait_neutral:
            self.reset_targets()
            self._wait_neutral = not self._is_neutral(inputs)
            self._last_update = time.perf_counter()

            return self.mapper(
                *self.integrator.get_targets()
            )

        dt = (
            0.0
            if self._last_update is None
            else now - self._last_update
        )
        self._last_update = now



        # 각 마우스의 1번 버튼으로 위치+회전 / 회전만 전환한다.
        raw_inputs = {side: dict(raw) for side, raw in inputs.items()}
        inputs = {side: dict(raw) for side, raw in inputs.items()}
        for side in ("left", "right"):
            raw = inputs[side]
            buttons = raw["buttons"]
            index = self.config.mode_toggle_button
            pressed = index < len(buttons) and bool(buttons[index])
            if pressed and not self._mode_button_pressed[side]:
                self._rotation_only[side] = not self._rotation_only[side]
                self._mode_wait_neutral[side] = True
                name = "회전만" if self._rotation_only[side] else "위치 + 회전"
                print(f"[MODE] {side}: {name}", flush=True)
            self._mode_button_pressed[side] = pressed

            if self._mode_wait_neutral[side]:
                axes = np.array([float(raw[k]) for k in AXES])
                if not np.isfinite(axes).all():
                    raise ValueError("Invalid SpaceMouse axes.")
                if not pressed and np.all(np.abs(axes) <= self.config.deadzone):
                    self._mode_wait_neutral[side] = False
                for axis in AXES:
                    raw[axis] = 0.0
            elif self._rotation_only[side]:
                for axis in ("x", "y", "z"):
                    raw[axis] = 0.0

        # 양쪽 마우스가 각각 움직임 입력 중인지 확인한다.
        active = {}

        for side in ("left", "right"):
            raw = inputs[side]
            sample_time = float(raw["t"])
            sample_age = now - sample_time

            sample_is_fresh = (
                np.isfinite(sample_time)
                and sample_time >= 0.0
                and 0.0 <= sample_age <= self.config.input_timeout_s
            )

            active[side] = sample_is_fresh and any(
                abs(float(raw[axis])) > self.config.deadzone
                for axis in ("x", "y", "z", "roll", "pitch", "yaw")
            )

        # 움직이다가 중립 상태가 되거나 입력이 오래된 팔을 찾는다.
        released_sides = [
            side
            for side in ("left", "right")
            if self._was_active[side] and not active[side]
        ]

        self._was_active = active



        poses, grippers = self.integrator.update(
            inputs,
            dt,
            now=now,
        )

        # 입력이 중립이 되거나 오래되면 적분은 중지하고 마지막 목표를 유지한다.
        # 실제 자세로 목표를 재설정하면 추종 오차가 목표에 누적될 수 있다.
        # 로봇은 남아 있는 추종 오차를 계속 따라갈 수 있다.
        if released_sides:
            print(
                "[MOUSE DEBUG] 입력 중립 또는 시간 초과: "
                f"{', '.join(released_sides)} 마지막 목표 유지",
                flush=True,
            )
            
        # 로봇이 목표를 따라가지 못할 때
        # 가상 목표가 계속 멀어지는 것을 제한한다.
        actual = self.state_reader.read()

        # 0.5초마다 마우스 입력, 목표 위치, 실제 위치 확인
        if now >= getattr(self, "_next_motion_debug", 0.0):
            raw = raw_inputs["left"]
            mode_input = inputs["left"]
            target_pose = poses["left"]
            actual_pose = actual["left"]
            target_xyz = target_pose[:3, 3]
            actual_xyz = actual_pose[:3, 3]
            # TCP는 설정한 arm_6 로컬 오프셋 기준의 추정 위치이다.
            target_tcp_xyz = target_xyz + target_pose[:3, :3] @ self.integrator.tcp_offset
            actual_tcp_xyz = actual_xyz + actual_pose[:3, :3] @ self.integrator.tcp_offset
            rotation_error = Rotation.from_matrix(
                actual_pose[:3, :3].T @ target_pose[:3, :3]
            ).magnitude()

            print(
                "[MOUSE DEBUG]",
                "raw_input=",
                {
                    key: round(float(raw[key]), 3)
                    for key in ("x", "y", "z", "roll", "pitch", "yaw")
                },
                "mode_input=",
                {
                    key: round(float(mode_input[key]), 3)
                    for key in ("x", "y", "z", "roll", "pitch", "yaw")
                },
                "input_age_s=", round(now - float(raw["t"]), 3),
                "mode=", "rotation_only" if self._rotation_only["left"] else "translation_rotation",
                "target_arm6_xyz=", np.round(target_xyz, 4),
                "actual_arm6_xyz=", np.round(actual_xyz, 4),
                "target_tcp_xyz=", np.round(target_tcp_xyz, 4),
                "actual_tcp_xyz=", np.round(actual_tcp_xyz, 4),
                "arm6_error_m=", round(
                    float(np.linalg.norm(target_xyz - actual_xyz)),
                    5,
                ),
                "tcp_error_m=", round(
                    float(np.linalg.norm(target_tcp_xyz - actual_tcp_xyz)),
                    5,
                ),
                "rotation_error_rad=", round(float(rotation_error), 5),
                flush=True,
            )

            self._next_motion_debug = now + 0.5



        for side in ("left", "right"):
            position_error = np.linalg.norm(
                poses[side][:3, 3]
                - actual[side][:3, 3]
            )

            rotation_error = Rotation.from_matrix(
                actual[side][:3, :3].T
                @ poses[side][:3, :3]
            ).magnitude()

            if (
                position_error > self.config.max_target_lead_m
                or rotation_error
                > self.config.max_target_rotation_rad
            ):
                self._initialized = False

                raise RuntimeError(
                    f"{side} target is too far from the actual pose. "
                    "Check robot tracking and reinitialize."
                )

        return self.mapper(poses, grippers)

    def calibrate(self):
        pass

    def configure(self):
        pass

    def send_feedback(self, feedback):
        if not self.is_connected:
            raise RuntimeError("Teleoperator is not connected.")

    def disconnect(self):
        try:
            if self._stack is not None:
                self._stack.close()
        finally:
            self._stack = None
            self._follower = None
            self.state_reader = None
            self._initialized = False
            self._wait_neutral = True
            self._last_update = None


def install_lerobot_bridge():
    """표준 LeRobot 루프 시작 시 이 teleoperator에 robot을 연결한다."""
    specs = (
        (
            "lerobot.scripts.lerobot_teleoperate",
            "teleop_loop",
            1,
            0,
        ),
        (
            "lerobot.scripts.lerobot_record",
            "record_loop",
            0,
            7,
        ),
    )

    def wrap_loop(original, robot_position, teleop_position):
        @wraps(original)
        def bridged(*args, **kwargs):
            robot = (
                kwargs["robot"]
                if "robot" in kwargs
                else args[robot_position]
            )

            teleop = (
                kwargs["teleop"]
                if "teleop" in kwargs
                else args[teleop_position]
                if len(args) > teleop_position
                else None
            )

            if isinstance(teleop, SpaceMouse4Rby1):
                teleop.bind_robot(robot)

            return original(*args, **kwargs)

        bridged._spacemouse4rby1_bridge = True
        return bridged

    for (
        module_name,
        function_name,
        robot_position,
        teleop_position,
    ) in specs:
        module = sys.modules.get(module_name)

        if module is None:
            continue

        original = getattr(module, function_name, None)

        if original is None or getattr(
            original,
            "_spacemouse4rby1_bridge",
            False,
        ):
            continue

        setattr(
            module,
            function_name,
            wrap_loop(
                original,
                robot_position,
                teleop_position,
            ),
        )