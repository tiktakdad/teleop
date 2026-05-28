# Copyright 2026 ROBOTIS / teleop custom task
"""Robust zero-dependency hot-reloading proxy for Isaac Lab retargeters."""

from __future__ import annotations

import importlib
import os
import sys
import traceback
from typing import Any
import torch

from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg


class HotReloadProxy(RetargeterBase):
    """실시간 코드 수정 반영을 위한 핫 리로드 프록시 래퍼 클래스.

    원본 리타게터 파일의 수정(Ctrl+S)을 자동으로 감지하여,
    시뮬레이터를 재시작하지 않고 메모리 상에서 해당 클래스만 0.1초 만에 새로 고침(Reload)합니다.
    코드에 문법 오류가 있을 경우 시뮬레이터가 꺼지지 않도록 예외 처리를 하며 기존 정상 인스턴스를 유지합니다.
    """

    def __init__(self, module_name: str, class_name: str, cfg: RetargeterCfg, *args, **kwargs):
        self._module_name = module_name
        self._class_name = class_name
        self._cfg = cfg
        self._args = args
        self._kwargs = kwargs

        # 1. 대상 모듈 임포트 및 파일 경로 획득
        self._module = importlib.import_module(module_name)
        self._file_path = self._module.__file__
        self._last_mtime = os.path.getmtime(self._file_path)

        # 2. 실제 리타게터 클래스 인스턴스 생성
        class_type = getattr(self._module, class_name)
        self._instance = class_type(cfg, *args, **kwargs)

        # 3. 부모 클래스 초기화
        super().__init__(cfg)

        print("\n" + "=" * 80, flush=True)
        print(f"🔥 [HotReloadProxy] '{class_name}' 프록시가 초기화되었습니다.", flush=True)
        print(f"👀 실시간 코드 수정 감지 대상: '{self._file_path}'", flush=True)
        print("=" * 80 + "\n", flush=True)

    def _check_and_reload(self) -> bool:
        """소스 파일의 수정 여부를 감지하고 변경 시 핫 리로드를 수행합니다."""
        try:
            current_mtime = os.path.getmtime(self._file_path)
            if current_mtime > self._last_mtime:
                # 감지 시간 즉시 업데이트 (중복 리로드 방지)
                self._last_mtime = current_mtime
                print("\n" + "🔄" * 40, flush=True)
                print(f"[HotReloadProxy] 코드 변경 감지됨: '{self._file_path}'. 리로드 중...", flush=True)

                # 모듈 언로드 및 재임포트
                if self._module_name in sys.modules:
                    importlib.reload(self._module)
                else:
                    self._module = importlib.import_module(self._module_name)

                # 새로운 클래스 타입으로 재설정 및 생성
                class_type = getattr(self._module, self._class_name)
                new_instance = class_type(self._cfg, *self._args, **self._kwargs)

                # 성공 시에만 기존 인스턴스 대체
                self._instance = new_instance
                print(f"✅ [HotReloadProxy] '{self._class_name}' 리로드 완료! 변경 사항이 실시간 반영되었습니다.", flush=True)
                print("🔄" * 40 + "\n", flush=True)
                return True
        except Exception as e:
            # 코드 에러(SyntaxError 등) 발생 시 크래시를 막고 에러 로그 출력 후 기존 객체 유지
            print("\n" + "❌" * 40, flush=True)
            print(f"[HotReloadProxy] '{self._class_name}' 리로드 중 에러 발생:", flush=True)
            print("-" * 80, flush=True)
            traceback.print_exc()
            print("-" * 80, flush=True)
            print("[HotReloadProxy] ⚠️ 시뮬레이터 중단을 막기 위해 이전 정상 작동 상태를 유지합니다. 코드를 수정해주세요.", flush=True)
            print("❌" * 40 + "\n", flush=True)
        return False

    def retarget(self, data: dict) -> torch.Tensor:
        """리타게팅 연산 실행 시 실시간 변경 사항을 체크한 후 위임합니다."""
        self._check_and_reload()
        return self._instance.retarget(data)

    def get_requirements(self) -> set[Any]:
        """장치 요구 사항을 위임합니다."""
        self._check_and_reload()
        if hasattr(self._instance, "get_requirements"):
            return self._instance.get_requirements()
        return super().get_requirements()

    def __getattr__(self, name: str) -> Any:
        """기타 모든 속성 및 메서드 호출을 실제 리타게터 인스턴스로 전달합니다."""
        self._check_and_reload()
        return getattr(self._instance, name)
