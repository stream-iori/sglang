"""Mini Scheduler demo for Week 1/2.

Run from repo root:
    python sglang-learning-docs/06_demo_scheduler.py
"""

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class ForwardMode(Enum):
    EXTEND = "extend"
    DECODE = "decode"


@dataclass
class Req:
    rid: str
    prompt_tokens: List[int]
    max_new_tokens: int = 4
    output_tokens: List[int] = field(default_factory=list)

    def finished(self) -> bool:
        return len(self.output_tokens) >= self.max_new_tokens


@dataclass
class ScheduleBatch:
    reqs: List[Req]
    forward_mode: ForwardMode


class FakeModel:
    def forward(self, batch: ScheduleBatch) -> List[int]:
        time.sleep(0.02)
        return [random.randint(100, 999) for _ in batch.reqs]


class MiniScheduler:
    def __init__(self):
        self.waiting_queue: List[Req] = []
        self.running_batch: List[Req] = []
        self.finished_reqs: List[Req] = []
        self.model = FakeModel()

    def add_request(self, req: Req):
        print(f"[recv] rid={req.rid}, prompt_len={len(req.prompt_tokens)}")
        self.waiting_queue.append(req)

    def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
        if self.waiting_queue:
            reqs = self.waiting_queue[:4]
            self.waiting_queue = self.waiting_queue[4:]
            self.running_batch.extend(reqs)
            return ScheduleBatch(reqs=reqs, forward_mode=ForwardMode.EXTEND)
        if self.running_batch:
            return ScheduleBatch(reqs=list(self.running_batch), forward_mode=ForwardMode.DECODE)
        return None

    def run_batch(self, batch: ScheduleBatch) -> List[int]:
        print(f"[run] mode={batch.forward_mode.value}, batch_size={len(batch.reqs)}")
        return self.model.forward(batch)

    def process_batch_result(self, batch: ScheduleBatch, next_tokens: List[int]):
        for req, token in zip(batch.reqs, next_tokens):
            req.output_tokens.append(token)
            print(f"[out] rid={req.rid}, token={token}, total={len(req.output_tokens)}")
            if req.finished() and req in self.running_batch:
                self.running_batch.remove(req)
                self.finished_reqs.append(req)
                print(f"[done] rid={req.rid}, output={req.output_tokens}")

    def event_loop_step(self) -> bool:
        batch = self.get_next_batch_to_run()
        if batch is None:
            print("[idle]")
            return False
        result = self.run_batch(batch)
        self.process_batch_result(batch, result)
        return True


def main():
    scheduler = MiniScheduler()
    scheduler.add_request(Req("req-1", [1, 2, 3], max_new_tokens=3))
    scheduler.add_request(Req("req-2", [10, 11], max_new_tokens=5))

    step = 0
    while scheduler.waiting_queue or scheduler.running_batch:
        step += 1
        print(f"\nstep={step} waiting={len(scheduler.waiting_queue)} running={len(scheduler.running_batch)}")
        scheduler.event_loop_step()

    print(f"\nfinished={len(scheduler.finished_reqs)}")


if __name__ == "__main__":
    main()

