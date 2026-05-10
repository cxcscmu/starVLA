"""Calvin benchmark — data config, embodiment tag, and mixtures.

Defines a Calvin-specific Franka config that includes the state modality
(needed by frameworks like QwenPI_v3 which feed proprio into the prompt).

Why not reuse `libero_franka`?
- The LIBERO data_registry's `Libero4in1DataConfig` deliberately comments out
  the state ModalityConfig ("ignore state modality for now since some datasets
  don't have state"). Using `libero_franka` for Calvin ABC->D therefore drops
  state at the dataloader layer, which crashes QwenPI_v3 in `_pack_sample`
  (np.concatenate of an empty state list).
"""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


class CalvinFrankaDataConfig:
    video_keys = [
        "video.primary_image",
        "video.wrist_image",
    ]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.pad",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))
    state_indices = list(range(-16, 0))

    def modality_config(self):
        return {
            "video":  ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state":  ModalityConfig(delta_indices=self.state_indices,        modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices,       modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "min_max",
                    "action.y": "min_max",
                    "action.z": "min_max",
                    "action.roll": "min_max",
                    "action.pitch": "min_max",
                    "action.yaw": "min_max",
                },
            ),
        ])


ROBOT_TYPE_CONFIG_MAP = {
    "calvin_franka": CalvinFrankaDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "calvin_franka": EmbodimentTag.FRANKA,
}

DATASET_NAMED_MIXTURES = {
    "calvin_task_ABC_D": [
        ("calvin_task_ABC_D", 1.0, "calvin_franka"),
    ],
}
