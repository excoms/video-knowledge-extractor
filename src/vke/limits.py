"""Every run passes through here. Limits duplicated across entry points do not
hold; one gate does."""
from __future__ import annotations

import random
import re
import shutil
import subprocess
import time
from dataclasses import dataclass

WARN_ABOVE_VIDEOS = 30          # warn + confirm, never a hard block
MIN_FREE_DISK = 1.2 * 1024**3   # abort cleanly rather than dying mid-download
MIN_PAUSE = 1.5                 # floor; cannot be set to 0


class Aborted(RuntimeError):
    """Raised when a guard stops the run. Progress is always checkpointed."""


@dataclass
class Limits:
    pause: float = 2.0
    jitter: float = 4.0
    max_videos: int = 0          # 0 = no cap
    min_free_disk: float = MIN_FREE_DISK
    assume_yes: bool = False

    def __post_init__(self) -> None:
        # A zero pause is how a well-behaved client becomes a problem for
        # someone else's servers. Politeness is not user-configurable to off.
        self.pause = max(float(self.pause), MIN_PAUSE)
        self.jitter = max(float(self.jitter), 0.0)

    # -- pacing ------------------------------------------------------------
    def nap(self, why: str = "") -> None:
        t = self.pause + random.uniform(0, self.jitter)
        if why:
            print(f"      pausing {t:.1f}s {why}", flush=True)
        time.sleep(t)

    def ytdlp_opts(self) -> dict:
        """Randomised pacing for yt-dlp. Jitter manages rate-limit thresholds;
        it does not disguise automation, and we do not pretend otherwise."""
        return {
            "sleep_interval": self.pause,
            "max_sleep_interval": self.pause + self.jitter,
            "sleep_requests": 1,
            "retries": 3,
            "concurrent_fragment_downloads": 1,
        }

    # -- resources ---------------------------------------------------------
    @staticmethod
    def free_disk() -> int:
        return shutil.disk_usage("/").free

    @staticmethod
    def swap_used_mb() -> float:
        try:
            out = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                                 capture_output=True, text=True, timeout=3).stdout
            m = re.search(r"used\s*=\s*([\d.]+)M", out)
            return float(m.group(1)) if m else -1.0
        except Exception:
            return -1.0

    def check_resources(self, stage: str) -> None:
        free = self.free_disk()
        if free < self.min_free_disk:
            raise Aborted(
                f"Only {free / 1024**3:.2f} GB free before {stage}; "
                f"{self.min_free_disk / 1024**3:.1f} GB needed.\n"
                f"Free some space and run the same command again — finished "
                f"videos are already saved and will be skipped."
            )

    # -- scale -------------------------------------------------------------
    def confirm_scale(self, n: int, est_seconds: float | None = None) -> None:
        if n <= WARN_ABOVE_VIDEOS or self.assume_yes:
            return
        est = ""
        if est_seconds:
            est = f"  Roughly {est_seconds / 60:.0f} min of transcription."
        print(f"\n  {n} videos.{est}")
        print("  Large runs take a while and make a lot of requests to YouTube.")
        try:
            if input("  Continue? [y/N] ").strip().lower() not in ("y", "yes"):
                raise Aborted("Stopped at the confirmation prompt.")
        except EOFError:
            raise Aborted("Needs confirmation for a run this size; pass --yes.")
