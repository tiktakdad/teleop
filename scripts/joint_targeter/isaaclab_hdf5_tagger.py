# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
IsaacLabHdf5Tagger - 범용 HDF5 메타데이터 태거

Isaac Lab 환경에서 수집된 HDF5 데이터셋에 관절/액션 메타데이터를 주입하고,
state-action 표현 방식이 불일치할 경우 자동으로 관절 공간으로 변환합니다.

Usage:
    from isaaclab_hdf5_tagger import IsaacLabHdf5Tagger

    # 녹화 종료 후, env.close() 전에 호출
    IsaacLabHdf5Tagger.tag_all(env, dataset_path, auto_align=True)
"""

from __future__ import annotations

import json
import os

import h5py
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# 메인 클래스
# ═══════════════════════════════════════════════════════════════════════════


class IsaacLabHdf5Tagger:
    """범용 HDF5 메타데이터 태거.

    모든 Isaac Lab 환경에서 로봇 종류에 관계없이 동작합니다.
    하드코딩 없이 IO descriptor를 통해 동적으로 이름과 표현 방식을 추출합니다.

    태깅하는 메타데이터 (HDF5 root attributes):
        robot_joint_names  : 로봇의 물리 관절 이름 (state 공간)
        action_names       : 액션 차원별 이름
        action_space_type  : 액션 표현 방식 ("joint_position" | "eef_pose" | "mixed" | ...)
        action_term_info   : 각 액션 텀의 상세 정보 (JSON 문자열)

    auto_align 변환 시 추가되는 메타데이터:
        original_action_dim : 변환 전 원래 액션 차원 (예: 32)
        conversion_source   : 변환에 사용된 observation 키 (예: "obs/robot_joint_pos[t+1]")
    """

    # ═════════════════════════════════════════════════════════════════════
    # 공개 API
    # ═════════════════════════════════════════════════════════════════════

    @staticmethod
    def tag_all(
        env,
        dataset_path: str,
        robot_name: str = "robot",
        auto_align: bool = False,
    ) -> dict:
        """한 번의 호출로 모든 메타데이터를 태깅합니다.

        진행 흐름:
            1. tag_joint_names()  → robot_joint_names 태깅
            2. tag_action_info()  → action_names, action_space_type 태깅
            3. (auto_align=True일 때만)
               → action_space_type이 "eef_pose" / "mixed" / "velocity_command"이면
               → convert_actions_to_joint_space() 호출
               → action[t] = obs/robot_joint_pos[t+1]로 대체

        Args:
            env: Isaac Lab 환경 객체 (gym.Env).
            dataset_path: 저장된 HDF5 파일 경로.
            robot_name: 씬 내 로봇 에셋 이름 (기본: "robot").
            auto_align: True이면 state/action 불일치 시 자동으로 관절 공간 변환.

        Returns:
            태깅 결과 요약 dict:
                {"joint_names": bool, "action": bool, "converted": bool | None}
        """
        result = {}
        result["joint_names"] = IsaacLabHdf5Tagger.tag_joint_names(
            env, dataset_path, robot_name
        )
        result["action"] = IsaacLabHdf5Tagger.tag_action_info(env, dataset_path)

        if auto_align:
            term_infos = IsaacLabHdf5Tagger._analyze_action_terms(env)
            space_types = set(info["space_type"] for info in term_infos)
            needs_conversion = (
                "eef_pose" in space_types
                or "mixed" in space_types
                or "velocity_command" in space_types
            )

            if needs_conversion:
                print("\n[Tagger] auto_align=True: action이 관절 공간이 아닙니다.")
                print("[Tagger]    obs/robot_joint_pos[t+1] -> action[t] 변환을 수행합니다.\n")
                result["converted"] = IsaacLabHdf5Tagger.convert_actions_to_joint_space(
                    dataset_path, robot_name
                )
            else:
                print("[Tagger] auto_align: action이 이미 관절 공간입니다. 변환 불필요.")
                result["converted"] = False
        else:
            result["converted"] = None

        return result

    # ═════════════════════════════════════════════════════════════════════
    # 1. 관절 이름 태깅 (state 공간)
    # ═════════════════════════════════════════════════════════════════════

    @staticmethod
    def tag_joint_names(
        env, dataset_path: str, robot_name: str = "robot"
    ) -> bool:
        """로봇의 물리 관절 이름을 HDF5 root attrs에 태깅합니다.

        추출 경로 (우선순위 순):
            1. env.scene[robot_name].joint_names
            2. env.unwrapped.scene[robot_name].joint_names
            3. env.<robot_name>.joint_names

        Args:
            env: Isaac Lab 환경 객체.
            dataset_path: HDF5 파일 경로.
            robot_name: 씬 내 로봇 에셋 이름.

        Returns:
            태깅 성공 여부.
        """
        if not os.path.exists(dataset_path):
            print(f"[Tagger] HDF5 파일 없음, 건너뜀: {dataset_path}")
            return False

        joint_names = IsaacLabHdf5Tagger._get_robot_joint_names(env, robot_name)
        if not joint_names:
            print(f"[Tagger] '{robot_name}' 관절 정보를 찾을 수 없습니다.")
            return False

        with h5py.File(dataset_path, "a") as f:
            f.attrs["robot_joint_names"] = np.array(joint_names, dtype="S")

        print(f"[Tagger] robot_joint_names ({len(joint_names)}개) 태깅 완료")
        return True

    # ═════════════════════════════════════════════════════════════════════
    # 2. 액션 정보 태깅 (이름 + 표현 방식 + 텀별 상세)
    # ═════════════════════════════════════════════════════════════════════

    @staticmethod
    def tag_action_info(env, dataset_path: str) -> bool:
        """액션 이름, 표현 방식, 텀별 상세 정보를 HDF5에 태깅합니다.

        분석 흐름:
            1. env.action_manager._terms 에서 모든 액션 텀을 순회
            2. 각 텀의 IO descriptor → action_type 으로 분류:
               - "JointAction"                   → joint_position
               - "TaskSpaceAction"                → eef_pose
               - "PinkInverseKinematicsAction"    → eef_pose
               - "non holonomic actions"          → velocity_command
               - io_desc.joint_names 존재         → joint_position
               - term._joint_names 존재           → joint_position
               - 모두 실패                         → unknown
            3. 모든 텀의 space_type을 종합:
               - 전부 같으면 → 해당 타입
               - 섞여 있으면 → "mixed"
            4. state 관절 이름과 비교해 일치 여부 리포트

        Args:
            env: Isaac Lab 환경 객체.
            dataset_path: HDF5 파일 경로.

        Returns:
            태깅 성공 여부.
        """
        if not os.path.exists(dataset_path):
            print(f"[Tagger] HDF5 파일 없음, 건너뜀: {dataset_path}")
            return False

        term_infos = IsaacLabHdf5Tagger._analyze_action_terms(env)
        if not term_infos:
            print("[Tagger] 액션 텀을 분석할 수 없습니다.")
            return False

        # 이름 목록 구성
        all_names = []
        for info in term_infos:
            all_names.extend(info["dim_names"])

        # 전체 액션 공간 타입 결정
        space_types = set(info["space_type"] for info in term_infos)
        if len(space_types) == 1:
            overall_type = space_types.pop()
        else:
            overall_type = "mixed"

        # state와의 일치 여부 확인
        state_joint_names = IsaacLabHdf5Tagger._get_robot_joint_names(env)
        consistency = IsaacLabHdf5Tagger._check_state_action_consistency(
            all_names, state_joint_names, overall_type
        )

        # HDF5에 저장
        with h5py.File(dataset_path, "a") as f:
            f.attrs["action_names"] = np.array(all_names, dtype="S")
            f.attrs["action_space_type"] = overall_type
            f.attrs["action_term_info"] = json.dumps(
                [_serialize_term_info(info) for info in term_infos], indent=2
            )

        # 결과 출력
        print(f"\n{'=' * 70}")
        print(f"[Tagger] 액션 메타데이터 태깅 완료")
        print(f"  총 액션 차원 : {len(all_names)}D")
        print(f"  표현 방식    : {overall_type}")
        print(f"  state 일치   : {consistency['message']}")
        print(f"{'─' * 70}")
        for info in term_infos:
            print(f"  [{info['term_name']}] {info['space_type']}, {info['action_dim']}D")
            for i, name in enumerate(info["dim_names"]):
                print(f"    [{i:2d}] {name}")
        print(f"{'=' * 70}\n")

        return True

    # ═════════════════════════════════════════════════════════════════════
    # 3. action → 관절 공간 자동 변환
    # ═════════════════════════════════════════════════════════════════════

    @staticmethod
    def convert_actions_to_joint_space(
        dataset_path: str,
        robot_name: str = "robot",
        obs_key: str = "robot_joint_pos",
        backup: bool = True,
        verbose: bool = True,
    ) -> bool:
        """HDF5의 action을 관절 공간으로 변환합니다.

        변환 원리:
            action[t]는 "시점 t에서 내린 명령"이고,
            obs/robot_joint_pos[t+1]은 "그 명령이 실행된 결과"입니다.
            따라서 action[t] = robot_joint_pos[t+1]로 대체하면
            action이 관절 공간 표현이 됩니다.

            마지막 프레임(t=T-1)은 다음 스텝이 없으므로 robot_joint_pos[T-1]을 유지합니다.

        변환 후 HDF5 변경:
            data/demo_X/actions         : (T, old_dim) → (T, joint_dim)
            data/demo_X/actions_original: (T, old_dim) 백업 (backup=True일 때)
            root attrs:
                action_names       → robot_joint_names와 동일
                action_space_type  → "joint_position"
                original_action_dim → 변환 전 차원 (예: 32)
                conversion_source   → "obs/robot_joint_pos[t+1]"

        Args:
            dataset_path: HDF5 파일 경로.
            robot_name: 로봇 에셋 이름 (메타데이터 기록용).
            obs_key: 관절 위치 observation 키 (기본: "robot_joint_pos").
            backup: True이면 변환 전 원본 action을 actions_original로 백업.

        Returns:
            성공 여부.
        """
        if not os.path.exists(dataset_path):
            print(f"[Tagger] 파일 없음: {dataset_path}")
            return False

        with h5py.File(dataset_path, "a") as f:
            data = f.get("data")
            if data is None:
                print("[Tagger] 'data' 그룹이 없습니다.")
                return False

            demo_keys = sorted(
                [k for k in data.keys() if k.startswith("demo_")],
                key=lambda x: int(x.split("_")[1]),
            )
            if not demo_keys:
                print("[Tagger] 데모 에피소드가 없습니다.")
                return False

            demo0 = data[demo_keys[0]]
            if "actions" not in demo0:
                print("[Tagger] 'actions' 데이터가 없습니다.")
                return False
            if obs_key not in demo0["obs"]:
                print(f"[Tagger] obs/{obs_key} 가 없습니다. 변환 불가.")
                return False

            joint_dim = demo0["obs"][obs_key].shape[1]
            num_demos = len(demo_keys)

            # 멱등성: 변환 전 원래 차원은 기존 attr 또는 백업(actions_original)에서
            # 복원한다. 매 에피소드 export 직후 반복 호출해도 안전하도록 한다.
            if "original_action_dim" in f.attrs:
                old_action_dim = int(f.attrs["original_action_dim"])
            elif "actions_original" in demo0:
                old_action_dim = demo0["actions_original"].shape[1]
            else:
                old_action_dim = demo0["actions"].shape[1]

            if verbose:
                print(f"[Tagger] action 변환 시작")
                print(f"  변환 전: actions {old_action_dim}D")
                print(f"  변환 후: actions {joint_dim}D (obs/{obs_key}에서)")
                print(f"  대상: {num_demos}개 에피소드")

            # actions뿐 아니라 processed_actions도 함께 변환한다.
            # LeRobot 변환기는 action 소스로 processed_actions를 사용하므로,
            # 둘 다 관절 공간으로 맞춰야 state/action 이름·차원이 일치한다.
            # (각 데이터셋은 *_original 로 백업)
            action_dataset_keys = ["actions", "processed_actions"]
            converted_count = 0
            skipped_count = 0
            for demo_key in demo_keys:
                demo = data[demo_key]

                # 이미 변환된 데모(actions_original 존재)는 건너뛴다 → 멱등.
                if "actions_original" in demo:
                    skipped_count += 1
                    continue

                joint_pos = demo["obs"][obs_key][:]

                T = joint_pos.shape[0]

                new_actions = np.zeros((T, joint_dim), dtype=np.float32)
                if T > 1:
                    new_actions[:-1] = joint_pos[1:]
                    new_actions[-1] = joint_pos[-1]
                elif T == 1:
                    new_actions[0] = joint_pos[0]

                for akey in action_dataset_keys:
                    if akey not in demo:
                        continue
                    backup_key = f"{akey}_original"
                    if backup and backup_key not in demo:
                        demo.create_dataset(backup_key, data=demo[akey][:])
                    del demo[akey]
                    demo.create_dataset(akey, data=new_actions)

                if "num_samples" in demo.attrs:
                    demo.attrs["num_samples"] = T

                converted_count += 1

            # 메타데이터 업데이트
            if "robot_joint_names" in f.attrs:
                joint_names = [
                    n.decode() if isinstance(n, bytes) else str(n)
                    for n in f.attrs["robot_joint_names"]
                ]
            else:
                joint_names = [f"{obs_key}_{i}" for i in range(joint_dim)]

            f.attrs["action_names"] = np.array(joint_names, dtype="S")
            f.attrs["action_space_type"] = "joint_position"
            if "original_action_dim" not in f.attrs:
                f.attrs["original_action_dim"] = old_action_dim
            f.attrs["conversion_source"] = f"obs/{obs_key}[t+1]"

            converted_term_info = [{
                "term_name": "converted_joint_position",
                "term_class": f"auto_converted_from_{old_action_dim}D",
                "action_dim": joint_dim,
                "space_type": "joint_position",
                "dim_names": joint_names,
            }]
            f.attrs["action_term_info"] = json.dumps(converted_term_info, indent=2)

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"[Tagger] action -> 관절 공간 변환 완료")
            print(f"  {old_action_dim}D (EEF/혼합) -> {joint_dim}D (관절 위치)")
            print(f"  {converted_count}개 신규 변환 / {skipped_count}개 기변환(건너뜀) / 총 {num_demos}개")
            print(f"  원본 백업: {'actions_original에 저장' if backup else '없음'}")
            print(f"  action[t] = obs/{obs_key}[t+1]")
            print(f"{'=' * 70}\n")

        return True

    # ═════════════════════════════════════════════════════════════════════
    # 내부 헬퍼
    # ═════════════════════════════════════════════════════════════════════

    @staticmethod
    def _get_robot_joint_names(env, robot_name: str = "robot") -> list[str]:
        """env에서 로봇 관절 이름을 안전하게 추출합니다."""
        for source in [env, getattr(env, "unwrapped", None)]:
            if source is None:
                continue
            if hasattr(source, "scene"):
                try:
                    return list(source.scene[robot_name].joint_names)
                except Exception:
                    pass
            robot_asset = getattr(source, robot_name, None)
            if robot_asset and hasattr(robot_asset, "joint_names"):
                return list(robot_asset.joint_names)
        return []

    @staticmethod
    def _analyze_action_terms(env) -> list[dict]:
        """action_manager의 모든 텀을 IO descriptor 기반으로 분석합니다."""
        try:
            am = env.action_manager
            terms = getattr(am, "_terms", None)
            if not terms:
                return []
        except Exception:
            return []

        results = []
        for term_name, term in terms.items():
            info = IsaacLabHdf5Tagger._analyze_single_term(term_name, term)
            results.append(info)
            print(f"[Tagger] 분석: {term_name} -> {info['space_type']}, {info['action_dim']}D")

        return results

    @staticmethod
    def _analyze_single_term(term_name: str, term) -> dict:
        """단일 액션 텀을 분석합니다.

        IO descriptor의 action_type을 우선 사용하며, 7단계 분류를 거칩니다:

        분류 우선순위:
            Case 1: action_type == "JointAction"
                    → io_desc.joint_names로 이름 생성, space_type = "joint_position"
            Case 2: action_type == "TaskSpaceAction"
                    → body_name + dim으로 pos/quat 이름 생성, space_type = "eef_pose"
            Case 3: action_type == "PinkInverseKinematicsAction"
                    → eef_link_names + hand_joint_names, space_type = "eef_pose"
            Case 4: action_type에 "non holonomic" 포함
                    → vx/vy/wz 이름 생성, space_type = "velocity_command"
            Case 5: io_desc에 joint_names 속성 존재 (커스텀 텀)
                    → joint_names 사용, space_type = "joint_position"
            Case 6: term 자체에 _joint_names 속성 존재
                    → _joint_names 사용, space_type = "joint_position"
            Case 7: 모두 실패
                    → 제네릭 이름 (term_name_0, _1, ...), space_type = "unknown"
        """
        cls_name = type(term).__name__
        dim = term.action_dim

        io_desc = None
        try:
            io_desc = term.IO_descriptor
        except Exception:
            pass

        action_type = getattr(io_desc, "action_type", None) if io_desc else None

        # Case 1: JointAction 계열
        if action_type == "JointAction":
            joint_names = getattr(io_desc, "joint_names", None)
            if joint_names and len(joint_names) == dim:
                return _build_term_info(
                    term_name, cls_name, dim, "joint_position",
                    list(joint_names), io_desc,
                )

        # Case 2: TaskSpaceAction 계열
        if action_type == "TaskSpaceAction":
            body_name = getattr(io_desc, "body_name", term_name)
            names = _make_task_space_names(body_name, dim)
            return _build_term_info(
                term_name, cls_name, dim, "eef_pose", names, io_desc,
            )

        # Case 3: PinkInverseKinematicsAction
        if action_type == "PinkInverseKinematicsAction":
            names = _make_pink_ik_names(term)
            return _build_term_info(
                term_name, cls_name, dim, "eef_pose", names, io_desc,
            )

        # Case 4: NonHolonomicAction
        if action_type and "non holonomic" in action_type.lower():
            body_name = getattr(io_desc, "body_name", "base")
            names = [f"{body_name}_vx", f"{body_name}_vy", f"{body_name}_wz"][:dim]
            return _build_term_info(
                term_name, cls_name, dim, "velocity_command", names, io_desc,
            )

        # Case 5: IO descriptor에 joint_names가 있는 커스텀 텀
        joint_names = getattr(io_desc, "joint_names", None) if io_desc else None
        if joint_names and len(joint_names) == dim:
            return _build_term_info(
                term_name, cls_name, dim, "joint_position",
                list(joint_names), io_desc,
            )

        # Case 6: term._joint_names 속성
        term_joint_names = getattr(term, "_joint_names", None)
        if term_joint_names and len(term_joint_names) == dim:
            return _build_term_info(
                term_name, cls_name, dim, "joint_position",
                list(term_joint_names), io_desc,
            )

        # Case 7: 분류 불가
        print(
            f"[Tagger] '{term_name}' ({cls_name}): "
            "IO descriptor로 분류 불가, 제네릭 이름 사용"
        )
        return _build_term_info(
            term_name, cls_name, dim, "unknown",
            [f"{term_name}_{i}" for i in range(dim)], io_desc,
        )

    @staticmethod
    def _check_state_action_consistency(
        action_names: list[str],
        state_joint_names: list[str],
        action_space_type: str,
    ) -> dict:
        """state와 action의 표현 방식 일치 여부를 검사합니다."""
        if not state_joint_names:
            return {
                "consistent": None,
                "message": "state 관절 정보 없음 (비교 불가)",
            }

        if action_space_type == "joint_position":
            overlap = set(action_names) & set(state_joint_names)
            ratio = len(overlap) / len(action_names) if action_names else 0
            if ratio > 0.8:
                return {
                    "consistent": True,
                    "message": f"관절 공간 일치 ({len(overlap)}/{len(action_names)} 겹침)",
                }
            else:
                return {
                    "consistent": False,
                    "message": f"관절 이름 불일치 ({len(overlap)}/{len(action_names)} 겹침)",
                }

        elif action_space_type == "eef_pose":
            return {
                "consistent": False,
                "message": "action=EEF포즈, state=관절각도 -> 공간 불일치",
            }

        elif action_space_type == "mixed":
            return {
                "consistent": False,
                "message": "action=혼합(EEF+관절+기타), state=관절각도 -> 부분 불일치",
            }

        return {
            "consistent": None,
            "message": f"action_space_type={action_space_type} (자동 판별 불가)",
        }


# ═══════════════════════════════════════════════════════════════════════════
# 모듈 레벨 헬퍼 함수
# ═══════════════════════════════════════════════════════════════════════════


def _build_term_info(
    term_name: str,
    term_class: str,
    action_dim: int,
    space_type: str,
    dim_names: list[str],
    io_descriptor,
) -> dict:
    """액션 텀 분석 결과 dict를 생성합니다."""
    return {
        "term_name": term_name,
        "term_class": term_class,
        "action_dim": action_dim,
        "space_type": space_type,
        "dim_names": dim_names,
        "io_descriptor": io_descriptor,
    }


def _make_pink_ik_names(term) -> list[str]:
    """PinkInverseKinematicsAction 텀에서 차원 이름을 추출합니다."""
    cfg = term.cfg
    names = []
    for eef in cfg.target_eef_link_names.keys():
        names += [
            f"{eef}_pos_x", f"{eef}_pos_y", f"{eef}_pos_z",
            f"{eef}_quat_w", f"{eef}_quat_x", f"{eef}_quat_y", f"{eef}_quat_z",
        ]
    if hasattr(cfg, "hand_joint_names"):
        names += list(cfg.hand_joint_names)
    return names


def _make_task_space_names(body_name: str, dim: int) -> list[str]:
    """태스크 공간 액션의 차원 이름을 생성합니다.

    dim에 따라 자동 결정:
        3D  -> pos(x,y,z)
        6D  -> pos + euler(r,p,y)
        7D  -> pos + quat(w,x,y,z)
        기타 -> pos + rot_0, rot_1, ...
    """
    pos = [f"{body_name}_pos_x", f"{body_name}_pos_y", f"{body_name}_pos_z"]
    if dim == 3:
        return pos
    elif dim == 6:
        return pos + [f"{body_name}_rot_r", f"{body_name}_rot_p", f"{body_name}_rot_y"]
    elif dim == 7:
        return pos + [
            f"{body_name}_quat_w", f"{body_name}_quat_x",
            f"{body_name}_quat_y", f"{body_name}_quat_z",
        ]
    else:
        return pos + [f"{body_name}_rot_{i}" for i in range(dim - 3)]


def _serialize_term_info(info: dict) -> dict:
    """term_info를 JSON 직렬화 가능한 형태로 변환합니다."""
    return {
        "term_name": info["term_name"],
        "term_class": info["term_class"],
        "action_dim": info["action_dim"],
        "space_type": info["space_type"],
        "dim_names": info["dim_names"],
    }
